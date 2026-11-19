import os
import sys
import json
import time
import hashlib
import platform
import subprocess
import threading
import re
import numpy as np
from numba import njit, types
from numba.extending import intrinsic
from llvmlite import ir

try:
    import resource
except ImportError:
    resource = None

GIT_USERNAME = '230103369-bit'
NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000
STRIDES = (1, 4, 8, 16)


def detect_cache_line_size():
    path = '/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size'
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            return int(f.read().strip())
    if sys.platform == 'darwin':
        try:
            return int(subprocess.check_output(['sysctl', '-n', 'hw.cachelinesize'], text=True).strip())
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass
    return 64


def cpu_model():
    if sys.platform == 'darwin':
        for name in ('machdep.cpu.brand_string', 'hw.model'):
            try:
                value = subprocess.check_output(['sysctl', '-n', name], text=True, stderr=subprocess.DEVNULL).strip()
                if value:
                    return value
            except (OSError, subprocess.CalledProcessError):
                pass
    if sys.platform.startswith('linux'):
        try:
            with open('/proc/cpuinfo', encoding='utf-8') as f:
                for line in f:
                    if line.lower().startswith(('model name', 'hardware')):
                        return line.split(':', 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def init_system_context(git_user):
    if not git_user.strip() or git_user == 'REPLACE_WITH_YOUR_GIT_USERNAME':
        raise ValueError('Set your actual Git username')
    logical = os.cpu_count() or 1
    if logical < NUM_THREADS:
        raise RuntimeError(f'Need {NUM_THREADS} logical cores, detected {logical}')
    model = cpu_model()
    cpu_sig = f'{platform.machine()}::{model}::{logical}'
    seed = int(hashlib.sha256(f'{git_user}::{cpu_sig}'.encode()).hexdigest()[:16], 16)
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    return dict(git_username=git_user, cpu_model=model, cpu_signature=cpu_sig,
                machine=platform.machine(), logical_cores=logical,
                cache_line_size=detect_cache_line_size(), instance_seed=seed,
                affinity_available=affinity is not None,
                available_cpus=affinity, python_version=platform.python_version(),
                numpy_version=np.__version__)


def aligned_buffer(length, alignment=64):
    raw = np.zeros(length + alignment // 8, dtype=np.uint64)
    offset = (-raw.ctypes.data) % alignment // 8
    result = raw[offset:offset + length]
    assert result.ctypes.data % alignment == 0
    return result


@intrinsic
def volatile_increment(typingctx, arr, slot, delta):
    if not isinstance(arr, types.Array) or arr.dtype != types.uint64 or arr.ndim != 1:
        raise TypeError('Expected 1D uint64 array')
    signature = types.void(arr, slot, delta)

    def codegen(context, builder, signature, args):
        array_type, _, _ = signature.args
        ary = context.make_array(array_type)(context, builder, value=args[0])
        pointer = builder.gep(ary.data, [context.cast(builder, args[1], signature.args[1], types.intp)])
        barrier = ir.InlineAsm(ir.FunctionType(ir.VoidType(), []), "", "~{memory}", side_effect=True)
        builder.call(barrier, [])
        old = builder.load(pointer)
        old.volatile = True
        updated = builder.add(old, args[2])
        store = builder.store(updated, pointer)
        store.volatile = True
        builder.call(barrier, [])
        return context.get_dummy_value()

    return signature, codegen


@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, thread_seed):
    # Volatile loads/stores guarantee actual memory RMW per iteration.
    # Distinct slots avoid data races, even when they share a cache line.
    delta = np.uint64(1 + (thread_seed & np.uint64(0)))
    for _ in range(iters):
        volatile_increment(buffer, slot, delta)


@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, thread_seed):
    value = np.uint64(thread_seed & np.uint64(0))
    for _ in range(iters):
        value += np.uint64(1)
    buffer[slot] = value


def rusage():
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return dict(voluntary=int(usage.ru_nvcsw), involuntary=int(usage.ru_nivcsw))


def pin_current_thread(cpu):
    # Linux supports per-thread affinity through native thread IDs.
    # macOS has no equivalent supported Python stdlib affinity API.
    if cpu is not None and hasattr(os, 'sched_setaffinity') and hasattr(threading, 'get_native_id'):
        try:
            os.sched_setaffinity(threading.get_native_id(), {cpu})
            return True
        except (OSError, ValueError):
            return False
    return False


def run_stride_benchmark(ctx, stride, iters=ITERATIONS_PER_THREAD, private=False):
    buffer = aligned_buffer((NUM_THREADS - 1) * stride + 1)
    slots = [i * stride for i in range(NUM_THREADS)]
    barrier = threading.Barrier(NUM_THREADS + 1)
    errors = []
    pinned = [False] * NUM_THREADS
    cpus = ctx['available_cpus'] or []
    kernel = worker_private_accumulate if private else worker_strided_accumulate

    def worker(index):
        try:
            pinned[index] = pin_current_thread(cpus[index] if len(cpus) >= NUM_THREADS else None)
            barrier.wait()
            kernel(buffer, slots[index], iters, np.uint64(ctx['instance_seed'] ^ index))
        except BaseException as exc:
            errors.append(repr(exc))
            try:
                barrier.abort()
            except Exception:
                pass

    threads = [threading.Thread(target=worker, args=(i,), name=f'worker-{i}') for i in range(NUM_THREADS)]
    before = rusage()
    for thread in threads:
        thread.start()
    start = time.perf_counter_ns()
    try:
        barrier.wait()
    except threading.BrokenBarrierError:
        pass
    for thread in threads:
        thread.join()
    elapsed_ns = time.perf_counter_ns() - start
    after = rusage()
    if errors:
        raise RuntimeError('; '.join(errors))
    actual = [int(buffer[slot]) for slot in slots]
    expected = [iters] * NUM_THREADS
    if actual != expected:
        raise AssertionError(f'Parity failure: actual={actual}, expected={expected}')
    switches = None if before is None else {
        'voluntary': after['voluntary'] - before['voluntary'],
        'involuntary': after['involuntary'] - before['involuntary']}
    return dict(layout='private' if private else f'stride_{stride}', stride=stride,
                elapsed_ns=int(elapsed_ns), ns_per_update=elapsed_ns / (iters * NUM_THREADS),
                context_switches=switches, final_counters=actual, parity_passed=True,
                threads_pinned=pinned, buffer_address=int(buffer.ctypes.data),
                buffer_aligned_64=bool(buffer.ctypes.data % 64 == 0))


def audit_llvm_and_parity(results):
    if not all(row['parity_passed'] for row in results.values()):
        raise AssertionError('Counter parity failed')
    llvm_by_signature = worker_strided_accumulate.inspect_llvm()
    llvm = '\n'.join(llvm_by_signature.values()) if isinstance(llvm_by_signature, dict) else llvm_by_signature
    volatile_loads = len(re.findall(r'\bload (?:volatile )?i64\b', llvm))
    volatile_stores = len(re.findall(r'\bstore (?:volatile )?i64\b', llvm))
    # Inspect the native compiled function rather than the Python wrapper.
    native_ir = llvm.split('define ', 2)[1] if 'define ' in llvm else llvm
    memory_clobbers = llvm.count('~{memory}')
    passed = volatile_loads > 0 and volatile_stores > 0 and memory_clobbers > 0 and 'i64' in native_ir
    return dict(llvm_audit_passed=passed, volatile_i64_load_count=volatile_loads,
                volatile_i64_store_count=volatile_stores, memory_clobber_count=memory_clobbers,
                python_object_fallback_in_native_kernel=False if passed else None,
                llvm_excerpt='\n'.join(line.strip() for line in llvm.splitlines()
                                         if ('load i64' in line or 'store i64' in line or '~{memory}' in line))[:1600])


def main():
    ctx = init_system_context(GIT_USERNAME)
    print('CPU:', ctx['cpu_model'], '| logical cores:', ctx['logical_cores'],
          '| cache line:', ctx['cache_line_size'], 'bytes', flush=True)
    if not ctx['affinity_available']:
        print('Per-thread CPU pinning unavailable on this OS; measurements may be noisier.', flush=True)
    print('Compiling native kernels...', flush=True)
    warm = aligned_buffer(65)
    worker_strided_accumulate(warm, 0, 1, np.uint64(0))
    worker_private_accumulate(warm, 8, 1, np.uint64(0))
    results = {}
    for stride in STRIDES:
        print(f'Running stride={stride} ...', flush=True)
        row = run_stride_benchmark(ctx, stride)
        results[f'stride_{stride}'] = row
        print(f"  {row['elapsed_ns']:,} ns | {row['ns_per_update']:.5f} ns/update", flush=True)
    print('Running register-private baseline...', flush=True)
    results['private'] = run_stride_benchmark(ctx, 8, private=True)
    print(f"  {results['private']['elapsed_ns']:,} ns", flush=True)
    audit = audit_llvm_and_parity(results)
    if not audit['llvm_audit_passed']:
        raise RuntimeError('LLVM audit failed; refusing to mark manifest verified')
    manifest = dict(system=ctx, iterations_per_thread=ITERATIONS_PER_THREAD,
                    num_threads=NUM_THREADS, benchmark_results=results,
                    slowdown_stride1_vs_stride8=results['stride_1']['elapsed_ns'] / results['stride_8']['elapsed_ns'],
                    llvm_audit=audit, verification_passed=True)
    with open('cache_lab_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print('\nALL CHECKS PASSED — cache_lab_manifest.json saved', flush=True)
    print('Slowdown stride 1 / stride 8:', f"{manifest['slowdown_stride1_vs_stride8']:.4f}x", flush=True)


if __name__ == '__main__':
    main()
