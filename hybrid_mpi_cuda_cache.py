import os
import sys
import hashlib
import platform
import subprocess

import numpy as np
from mpi4py import MPI
from numba import cuda

GITHUB_USERNAME = "230103369-bit"

def init_hybrid_context(github_user):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 4:
        raise RuntimeError("Exactly 4 MPI ranks are required")

    cart_comm = comm.Create_cart(
        dims=[2, 2],
        periods=[False, False],
        reorder=False
    )

    up, down = cart_comm.Shift(0, 1)
    left, right = cart_comm.Shift(1, 1)

    cuda.select_device(0)

    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_rank = node_comm.Get_rank()
    node_size = node_comm.Get_size()

    if node_size != 4:
        raise RuntimeError("All 4 ranks must share one node")

    # Rank 0 owns the shared allocation.
    # Extra space permits explicit 64-byte alignment.
    nbytes = 64 * np.dtype(np.uint64).itemsize + 63
    win = MPI.Win.Allocate_shared(
        nbytes if node_rank == 0 else 0,
        1,
        comm=node_comm
    )

    raw, disp_unit = win.Shared_query(0)
    base = np.frombuffer(raw, dtype=np.uint8)

    address = base.ctypes.data
    offset = (-address) % 64

    shm_view = np.frombuffer(
        raw,
        dtype=np.uint64,
        count=64,
        offset=offset
    )

    assert shm_view.ctypes.data % 64 == 0

    gpu_uuid = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True
    ).strip().splitlines()[0]

    cpu_signature = f"{platform.machine()}::{os.cpu_count()}"

    seed_material = (
        f"{github_user}::{cpu_signature}::{gpu_uuid}"
    )

    instance_seed = int(
        hashlib.sha256(seed_material.encode()).hexdigest()[:16],
        16
    )

    return {
        "comm": comm,
        "cart_comm": cart_comm,
        "node_comm": node_comm,
        "win": win,
        "shm_view": shm_view,
        "rank": rank,
        "size": size,
        "coords": cart_comm.Get_coords(rank),
        "neighbors": {
            "up": up,
            "down": down,
            "left": left,
            "right": right
        },
        "gpu_uuid": gpu_uuid,
        "cpu_signature": cpu_signature,
        "instance_seed": instance_seed
    }


if __name__ == "__main__":
    ctx = init_hybrid_context(GITHUB_USERNAME)

    for r in range(4):
        ctx["comm"].Barrier()

        if ctx["rank"] == r:
            print(
                f"Rank {r} | "
                f"Coords {ctx['coords']} | "
                f"Neighbors {ctx['neighbors']} | "
                f"Shared aligned: "
                f"{ctx['shm_view'].ctypes.data % 64 == 0}",
                flush=True
            )

    ctx["comm"].Barrier()

    if ctx["rank"] == 0:
        print("TASK 1 PASSED", flush=True)

    ctx["win"].Free()
    ctx["node_comm"].Free()
    ctx["cart_comm"].Free()


# ============================================================
# TASK 2 — GPU FIELD INITIALIZATION AND HALO PACK/UNPACK
# ============================================================

from numba import float64

LOCAL_N = 256
PADDED_N = LOCAL_N + 2


def initialize_fields(coords):
    row_rank, col_rank = coords

    u = np.zeros((PADDED_N, PADDED_N), dtype=np.float64)
    kappa = np.zeros_like(u)
    source = np.zeros_like(u)

    dx = 1.0 / 513.0

    for i in range(1, LOCAL_N + 1):
        for j in range(1, LOCAL_N + 1):
            global_i = row_rank * LOCAL_N + i
            global_j = col_rank * LOCAL_N + j

            x = global_j * dx
            y = global_i * dx

            u[i, j] = np.sin(np.pi * x) * np.sin(np.pi * y)
            kappa[i, j] = 0.8 + 0.2 * np.sin(np.pi * x) ** 2
            source[i, j] = 0.01 * np.cos(np.pi * x) * np.cos(np.pi * y)

    return u, kappa, source


@cuda.jit
def pack_halos_kernel(field, top, bottom, left, right):
    idx = cuda.grid(1)

    if idx < LOCAL_N:
        j = idx + 1

        top[idx] = field[1, j]
        bottom[idx] = field[LOCAL_N, j]

        left[idx] = field[j, 1]
        right[idx] = field[j, LOCAL_N]


@cuda.jit
def unpack_halos_kernel(field, top, bottom, left, right):
    idx = cuda.grid(1)

    if idx < LOCAL_N:
        j = idx + 1

        field[0, j] = top[idx]
        field[LOCAL_N + 1, j] = bottom[idx]

        field[j, 0] = left[idx]
        field[j, LOCAL_N + 1] = right[idx]


def test_task2(ctx):
    u, kappa, source = initialize_fields(ctx["coords"])

    d_u = cuda.to_device(u)

    send = [
        cuda.device_array(LOCAL_N, dtype=np.float64)
        for _ in range(4)
    ]

    recv = [
        cuda.to_device(np.full(LOCAL_N, float(i + 1)))
        for i in range(4)
    ]

    threads = 128
    blocks = (LOCAL_N + threads - 1) // threads

    pack_halos_kernel[blocks, threads](d_u, *send)
    cuda.synchronize()

    packed = [buf.copy_to_host() for buf in send]

    assert np.allclose(packed[0], u[1, 1:257])
    assert np.allclose(packed[1], u[256, 1:257])
    assert np.allclose(packed[2], u[1:257, 1])
    assert np.allclose(packed[3], u[1:257, 256])

    unpack_halos_kernel[blocks, threads](d_u, *recv)
    cuda.synchronize()

    result = d_u.copy_to_host()

    assert np.all(result[0, 1:257] == 1.0)
    assert np.all(result[257, 1:257] == 2.0)
    assert np.all(result[1:257, 0] == 3.0)
    assert np.all(result[1:257, 257] == 4.0)

    return {
        "rank": ctx["rank"],
        "field_shape": list(u.shape),
        "kappa_min": float(kappa[1:257, 1:257].min()),
        "kappa_max": float(kappa[1:257, 1:257].max()),
        "pack_passed": True,
        "unpack_passed": True,
    }


# ============================================================
# TASK 3 — 16x16 SHARED-MEMORY TILED STENCIL
# ============================================================

TILE_DIM = 16
SHARED_DIM = 18
LAMBDA_DAMP = 0.15


@cuda.jit
def tiled_stencil_kernel(u, kappa, source, out, dt, dx,
                         row_begin, row_end, col_begin, col_end):
    tile = cuda.shared.array((18, 18), dtype=float64)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    row = cuda.blockIdx.y * 16 + ty + 1
    col = cuda.blockIdx.x * 16 + tx + 1

    sy = ty + 1
    sx = tx + 1

    # Center
    tile[sy, sx] = u[row, col]

    # Four halo sides
    if tx == 0:
        tile[sy, 0] = u[row, col - 1]

    if tx == 15:
        tile[sy, 17] = u[row, col + 1]

    if ty == 0:
        tile[0, sx] = u[row - 1, col]

    if ty == 15:
        tile[17, sx] = u[row + 1, col]

    cuda.syncthreads()

    if (row_begin <= row <= row_end and
        col_begin <= col <= col_end and
        row <= 256 and col <= 256):

        center = tile[sy, sx]

        laplacian = (
            tile[sy - 1, sx] +
            tile[sy + 1, sx] +
            tile[sy, sx - 1] +
            tile[sy, sx + 1] -
            4.0 * center
        ) / (dx * dx)

        out[row, col] = (
            center +
            dt * (
                kappa[row, col] * laplacian +
                source[row, col] -
                LAMBDA_DAMP * center
            )
        )


def test_tiled_stencil(ctx):
    u, kappa, source = initialize_fields(ctx["coords"])

    d_u = cuda.to_device(u)
    d_kappa = cuda.to_device(kappa)
    d_source = cuda.to_device(source)
    d_out = cuda.to_device(u.copy())

    dx = 1.0 / 513.0
    kappa_max = float(kappa.max())
    dt_max = dx * dx / (4.0 * kappa_max)
    dt = 0.5 * dt_max

    blocks = (16, 16)
    threads = (16, 16)

    tiled_stencil_kernel[blocks, threads](
        d_u, d_kappa, d_source, d_out,
        dt, dx, 1, 256, 1, 256
    )

    cuda.synchronize()

    result = d_out.copy_to_host()

    assert np.isfinite(result).all()
    assert not np.array_equal(result[1:257, 1:257],
                              u[1:257, 1:257])

    # Reference check for an interior point.
    i, j = 128, 128

    lap = (
        u[i - 1, j] + u[i + 1, j] +
        u[i, j - 1] + u[i, j + 1] -
        4.0 * u[i, j]
    ) / (dx * dx)

    expected = u[i, j] + dt * (
        kappa[i, j] * lap +
        source[i, j] -
        LAMBDA_DAMP * u[i, j]
    )

    error = abs(result[i, j] - expected)
    assert error < 1e-12, f"Stencil mismatch: {error}"

    ptx = tiled_stencil_kernel.inspect_asm()
    asm = "\n".join(ptx.values()) if isinstance(ptx, dict) else ptx

    import re

    audit = {
        "st_shared_f64": len(re.findall(r"\bst\.shared\.f64\b", asm)),
        "ld_shared_f64": len(re.findall(r"\bld\.shared\.f64\b", asm)),
        "bar_sync": len(re.findall(r"\bbar\.sync\b", asm)),
    }

    print(
        f"Rank {ctx['rank']} | "
        f"Stencil error: {error:.3e} | "
        f"PTX audit: {audit}",
        flush=True
    )

    assert audit["bar_sync"] > 0, "Missing CUDA synchronization"
    assert audit["st_shared_f64"] > 0, "Missing shared-memory stores"
    assert audit["ld_shared_f64"] > 0, "Missing shared-memory loads"

    return audit


# ============================================================
# TASK 3B — MPI HALO EXCHANGE + DUAL CUDA STREAMS
# ============================================================

import time

DIRECTIONS = ("up", "down", "left", "right")

SEND_TAGS = {
    "up": 10,
    "down": 11,
    "left": 12,
    "right": 13,
}

OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


def create_pipeline(ctx):
    u, kappa, source = initialize_fields(ctx["coords"])

    dx = 1.0 / 513.0

    local_max = float(kappa[1:257, 1:257].max())
    kappa_max = ctx["cart_comm"].allreduce(
        local_max, op=MPI.MAX
    )

    dt_max = dx * dx / (4.0 * kappa_max)
    dt = 0.5 * dt_max

    stream_interior = cuda.stream()
    stream_boundary = cuda.stream()

    return {
        "u": cuda.to_device(u),
        "out": cuda.to_device(u.copy()),
        "kappa": cuda.to_device(kappa),
        "source": cuda.to_device(source),
        "dx": dx,
        "dt": dt,
        "dt_max": dt_max,
        "kappa_max": kappa_max,
        "stream_interior": stream_interior,
        "stream_boundary": stream_boundary,
        "send_gpu": {
            direction: cuda.device_array(
                256, dtype=np.float64
            )
            for direction in DIRECTIONS
        },
        "recv_gpu": {
            direction: cuda.device_array(
                256, dtype=np.float64
            )
            for direction in DIRECTIONS
        },
        "send_host": {
            direction: cuda.pinned_array(
                256, dtype=np.float64
            )
            for direction in DIRECTIONS
        },
        "recv_host": {
            direction: cuda.pinned_array(
                256, dtype=np.float64
            )
            for direction in DIRECTIONS
        },
    }


def launch_stencil(pipeline, stream, region):
    if region == "interior":
        bounds = (2, 255, 2, 255)
    elif region == "border":
        bounds = (1, 256, 1, 256)
    else:
        raise ValueError(region)

    if region == "interior":
        tiled_stencil_kernel[(16, 16), (16, 16), stream](
            pipeline["u"],
            pipeline["kappa"],
            pipeline["source"],
            pipeline["out"],
            pipeline["dt"],
            pipeline["dx"],
            *bounds
        )
    else:
        border_stencil_kernel[(16, 16), (16, 16), stream](
            pipeline["u"],
            pipeline["kappa"],
            pipeline["source"],
            pipeline["out"],
            pipeline["dt"],
            pipeline["dx"]
        )


@cuda.jit
def border_stencil_kernel(u, kappa, source, out, dt, dx):
    tile = cuda.shared.array((18, 18), dtype=float64)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    row = cuda.blockIdx.y * 16 + ty + 1
    col = cuda.blockIdx.x * 16 + tx + 1

    sy = ty + 1
    sx = tx + 1

    tile[sy, sx] = u[row, col]

    if tx == 0:
        tile[sy, 0] = u[row, col - 1]
    if tx == 15:
        tile[sy, 17] = u[row, col + 1]
    if ty == 0:
        tile[0, sx] = u[row - 1, col]
    if ty == 15:
        tile[17, sx] = u[row + 1, col]

    cuda.syncthreads()

    if row == 1 or row == 256 or col == 1 or col == 256:
        center = tile[sy, sx]

        lap = (
            tile[sy - 1, sx] +
            tile[sy + 1, sx] +
            tile[sy, sx - 1] +
            tile[sy, sx + 1] -
            4.0 * center
        ) / (dx * dx)

        out[row, col] = center + dt * (
            kappa[row, col] * lap +
            source[row, col] -
            LAMBDA_DAMP * center
        )


def exchange_halos(ctx, pipeline, overlapped):
    cart = ctx["cart_comm"]
    neighbors = ctx["neighbors"]

    boundary_stream = pipeline["stream_boundary"]
    interior_stream = pipeline["stream_interior"]

    requests = []

    # Post all receives first.
    for direction in DIRECTIONS:
        neighbor = neighbors[direction]

        if neighbor != MPI.PROC_NULL:
            req = cart.Irecv(
                pipeline["recv_host"][direction],
                source=neighbor,
                tag=SEND_TAGS[OPPOSITE[direction]]
            )
            requests.append(req)
        else:
            pipeline["recv_host"][direction].fill(0.0)

    # Pack outgoing halos on the GPU.
    pack_halos_kernel[2, 128, boundary_stream](
        pipeline["u"],
        pipeline["send_gpu"]["up"],
        pipeline["send_gpu"]["down"],
        pipeline["send_gpu"]["left"],
        pipeline["send_gpu"]["right"]
    )

    for direction in DIRECTIONS:
        if neighbors[direction] != MPI.PROC_NULL:
            pipeline["send_gpu"][direction].copy_to_host(
                pipeline["send_host"][direction],
                stream=boundary_stream
            )

    # MPI cannot send a buffer before its D2H transfer finishes.
    boundary_stream.synchronize()

    for direction in DIRECTIONS:
        neighbor = neighbors[direction]

        if neighbor != MPI.PROC_NULL:
            req = cart.Isend(
                pipeline["send_host"][direction],
                dest=neighbor,
                tag=SEND_TAGS[direction]
            )
            requests.append(req)

    # The deep interior does not depend on incoming halos.
    if overlapped:
        launch_stencil(pipeline, interior_stream, "interior")

    MPI.Request.Waitall(requests)

    # Copy received host buffers to GPU.
    for direction in DIRECTIONS:
        pipeline["recv_gpu"][direction].copy_to_device(
            pipeline["recv_host"][direction],
            stream=boundary_stream
        )

    unpack_halos_kernel[2, 128, boundary_stream](
        pipeline["u"],
        pipeline["recv_gpu"]["up"],
        pipeline["recv_gpu"]["down"],
        pipeline["recv_gpu"]["left"],
        pipeline["recv_gpu"]["right"]
    )

    if not overlapped:
        boundary_stream.synchronize()
        launch_stencil(pipeline, interior_stream, "interior")
        interior_stream.synchronize()

    # Border calculation waits for incoming halo unpacking.
    launch_stencil(pipeline, boundary_stream, "border")

    # Both streams must finish before swapping fields.
    interior_stream.synchronize()
    boundary_stream.synchronize()

    pipeline["u"], pipeline["out"] = (
        pipeline["out"], pipeline["u"]
    )


def step_serialized(ctx, pipeline):
    exchange_halos(ctx, pipeline, overlapped=False)


def step_overlapped(ctx, pipeline):
    exchange_halos(ctx, pipeline, overlapped=True)


def test_dual_stream(ctx):
    serial = create_pipeline(ctx)
    overlap = create_pipeline(ctx)

    cart = ctx["cart_comm"]

    cart.Barrier()
    start = time.perf_counter()

    for _ in range(3):
        step_serialized(ctx, serial)

    cart.Barrier()
    serial_ms = (time.perf_counter() - start) * 1000

    cart.Barrier()
    start = time.perf_counter()

    for _ in range(3):
        step_overlapped(ctx, overlap)

    cart.Barrier()
    overlap_ms = (time.perf_counter() - start) * 1000

    a = serial["u"].copy_to_host()
    b = overlap["u"].copy_to_host()

    local_diff = float(
        np.max(np.abs(a[1:257, 1:257] - b[1:257, 1:257]))
    )

    max_diff = cart.allreduce(local_diff, op=MPI.MAX)

    assert max_diff < 1e-12, (
        f"Serialized/overlapped mismatch: {max_diff}"
    )

    max_serial_ms = cart.allreduce(serial_ms, op=MPI.MAX)
    max_overlap_ms = cart.allreduce(overlap_ms, op=MPI.MAX)

    if ctx["rank"] == 0:
        print("DUAL STREAM TEST RESULTS")
        print(f"Serialized: {max_serial_ms:.3f} ms")
        print(f"Overlapped: {max_overlap_ms:.3f} ms")
        print(f"Maximum difference: {max_diff:.3e}")
        print("TASK 3B PASSED", flush=True)

    return {
        "serialized_ms": max_serial_ms,
        "overlapped_ms": max_overlap_ms,
        "max_diff": max_diff
    }


# ============================================================
# TASK 3C — WARMED 300-STEP BENCHMARK
# ============================================================

def benchmark_300_steps(ctx):
    cart = ctx["cart_comm"]
    rank = ctx["rank"]

    serial = create_pipeline(ctx)
    overlap = create_pipeline(ctx)

    # Compile all CUDA kernels before benchmarking.
    warmup_serial = create_pipeline(ctx)
    warmup_overlap = create_pipeline(ctx)

    step_serialized(ctx, warmup_serial)
    step_overlapped(ctx, warmup_overlap)

    cuda.synchronize()
    cart.Barrier()

    def run_mode(pipeline, step_function):
        cart.Barrier()
        start = time.perf_counter()

        for _ in range(300):
            step_function(ctx, pipeline)

        cuda.synchronize()
        cart.Barrier()

        local_ms = (time.perf_counter() - start) * 1000

        return cart.allreduce(local_ms, op=MPI.MAX)

    serialized_ms = run_mode(serial, step_serialized)
    overlapped_ms = run_mode(overlap, step_overlapped)

    a = serial["u"].copy_to_host()
    b = overlap["u"].copy_to_host()

    local_error = float(
        np.max(
            np.abs(
                a[1:257, 1:257] -
                b[1:257, 1:257]
            )
        )
    )

    max_error = cart.allreduce(local_error, op=MPI.MAX)

    local_energy_serial = float(
        np.sum(a[1:257, 1:257] ** 2)
    )

    local_energy_overlap = float(
        np.sum(b[1:257, 1:257] ** 2)
    )

    energy_serial = cart.allreduce(
        local_energy_serial, op=MPI.SUM
    )

    energy_overlap = cart.allreduce(
        local_energy_overlap, op=MPI.SUM
    )

    assert max_error < 1e-12, (
        f"Parity check failed: {max_error}"
    )

    speedup = serialized_ms / overlapped_ms

    if rank == 0:
        print("\nTASK 3C — 300-STEP BENCHMARK")
        print(f"Serialized: {serialized_ms:.3f} ms")
        print(f"Overlapped: {overlapped_ms:.3f} ms")
        print(f"Speedup: {speedup:.4f}x")
        print(f"Maximum error: {max_error:.3e}")
        print(f"Serialized energy: {energy_serial:.12f}")
        print(f"Overlapped energy: {energy_overlap:.12f}")
        print("TASK 3C PASSED", flush=True)

    return {
        "steps": 300,
        "serialized_ms": serialized_ms,
        "overlapped_ms": overlapped_ms,
        "speedup": speedup,
        "max_error": max_error,
        "serialized_energy": energy_serial,
        "overlapped_energy": energy_overlap,
        "verification_passed": max_error < 1e-12,
    }


# ============================================================
# TASK 4 — MPI FALSE SHARING BENCHMARK
# ============================================================

from numba import njit, types
from numba.extending import intrinsic
from llvmlite import ir
import time

CACHE_ITERATIONS = 40_000_000
CACHE_STRIDES = (1, 8, 16)


@intrinsic
def mpi_atomicity_free_increment(typingctx, array, index):
    sig = types.void(array, index)

    def codegen(context, builder, signature, args):
        array_value, index_value = args

        ary = context.make_array(signature.args[0])(
            context, builder, array_value
        )

        idx = context.cast(
            builder,
            index_value,
            signature.args[1],
            types.intp
        )

        pointer = builder.gep(ary.data, [idx])

        barrier = ir.InlineAsm(
            ir.FunctionType(ir.VoidType(), []),
            "",
            "~{memory}",
            side_effect=True
        )

        builder.call(barrier, [])

        current = builder.load(pointer)
        current.volatile = True

        updated = builder.add(
            current,
            ir.Constant(current.type, 1)
        )

        store = builder.store(updated, pointer)
        store.volatile = True

        builder.call(barrier, [])

        return context.get_dummy_value()

    return sig, codegen


@njit(nogil=True)
def mpi_false_sharing_worker(shared, slot, iterations):
    for _ in range(iterations):
        mpi_atomicity_free_increment(shared, slot)


def benchmark_mpi_false_sharing(ctx):
    comm = ctx["comm"]
    node_comm = ctx["node_comm"]
    win = ctx["win"]
    shared = ctx["shm_view"]
    rank = ctx["rank"]

    # JIT compilation before measurement.
    dummy = np.zeros(64, dtype=np.uint64)
    mpi_false_sharing_worker(dummy, 0, 1)

    results = {}

    for stride in CACHE_STRIDES:
        # Reset the shared counters.
        if rank == 0:
            shared[:] = 0

        win.Sync()
        node_comm.Barrier()
        win.Sync()

        slot = rank * stride

        comm.Barrier()
        start = time.perf_counter_ns()

        mpi_false_sharing_worker(
            shared, slot, CACHE_ITERATIONS
        )

        win.Sync()
        comm.Barrier()

        local_ns = time.perf_counter_ns() - start
        elapsed_ns = comm.allreduce(local_ns, op=MPI.MAX)

        win.Sync()
        node_comm.Barrier()
        win.Sync()

        if rank == 0:
            counts = [
                int(shared[r * stride])
                for r in range(4)
            ]

            passed = all(
                count == CACHE_ITERATIONS
                for count in counts
            )

            results[str(stride)] = {
                "stride": stride,
                "elapsed_ns": int(elapsed_ns),
                "elapsed_ms": elapsed_ns / 1e6,
                "counts": counts,
                "verification_passed": passed
            }

            print(
                f"Stride {stride:2d} | "
                f"{elapsed_ns / 1e6:.3f} ms | "
                f"Counts: {counts} | "
                f"PASS: {passed}",
                flush=True
            )

            if not passed:
                raise RuntimeError(
                    f"Counter verification failed for stride {stride}"
                )

        comm.Barrier()

    if rank == 0:
        t1 = results["1"]["elapsed_ns"]
        t8 = results["8"]["elapsed_ns"]
        t16 = results["16"]["elapsed_ns"]

        print("\nTASK 4 — SUMMARY")
        print(f"Stride 1 / 8 slowdown:  {t1 / t8:.4f}x")
        print(f"Stride 1 / 16 slowdown: {t1 / t16:.4f}x")
        print("TASK 4 PASSED", flush=True)

    return results if rank == 0 else None


# ============================================================
# TASK 4 — MPI FALSE SHARING BENCHMARK
# ============================================================

from numba import njit, types
from numba.extending import intrinsic
from llvmlite import ir
import time

CACHE_ITERATIONS = 40_000_000
CACHE_STRIDES = (1, 8, 16)


@intrinsic
def mpi_atomicity_free_increment(typingctx, array, index):
    sig = types.void(array, index)

    def codegen(context, builder, signature, args):
        array_value, index_value = args

        ary = context.make_array(signature.args[0])(
            context, builder, array_value
        )

        idx = context.cast(
            builder,
            index_value,
            signature.args[1],
            types.intp
        )

        pointer = builder.gep(ary.data, [idx])

        barrier = ir.InlineAsm(
            ir.FunctionType(ir.VoidType(), []),
            "",
            "~{memory}",
            side_effect=True
        )

        builder.call(barrier, [])

        current = builder.load(pointer)
        current.volatile = True

        updated = builder.add(
            current,
            ir.Constant(current.type, 1)
        )

        store = builder.store(updated, pointer)
        store.volatile = True

        builder.call(barrier, [])

        return context.get_dummy_value()

    return sig, codegen


@njit(nogil=True)
def mpi_false_sharing_worker(shared, slot, iterations):
    for _ in range(iterations):
        mpi_atomicity_free_increment(shared, slot)


def benchmark_mpi_false_sharing(ctx):
    comm = ctx["comm"]
    node_comm = ctx["node_comm"]
    win = ctx["win"]
    shared = ctx["shm_view"]
    rank = ctx["rank"]

    # JIT compilation before measurement.
    dummy = np.zeros(64, dtype=np.uint64)
    mpi_false_sharing_worker(dummy, 0, 1)

    results = {}

    for stride in CACHE_STRIDES:
        # Reset the shared counters.
        if rank == 0:
            shared[:] = 0

        win.Sync()
        node_comm.Barrier()
        win.Sync()

        slot = rank * stride

        comm.Barrier()
        start = time.perf_counter_ns()

        mpi_false_sharing_worker(
            shared, slot, CACHE_ITERATIONS
        )

        win.Sync()
        comm.Barrier()

        local_ns = time.perf_counter_ns() - start
        elapsed_ns = comm.allreduce(local_ns, op=MPI.MAX)

        win.Sync()
        node_comm.Barrier()
        win.Sync()

        if rank == 0:
            counts = [
                int(shared[r * stride])
                for r in range(4)
            ]

            passed = all(
                count == CACHE_ITERATIONS
                for count in counts
            )

            results[str(stride)] = {
                "stride": stride,
                "elapsed_ns": int(elapsed_ns),
                "elapsed_ms": elapsed_ns / 1e6,
                "counts": counts,
                "verification_passed": passed
            }

            print(
                f"Stride {stride:2d} | "
                f"{elapsed_ns / 1e6:.3f} ms | "
                f"Counts: {counts} | "
                f"PASS: {passed}",
                flush=True
            )

            if not passed:
                raise RuntimeError(
                    f"Counter verification failed for stride {stride}"
                )

        comm.Barrier()

    if rank == 0:
        t1 = results["1"]["elapsed_ns"]
        t8 = results["8"]["elapsed_ns"]
        t16 = results["16"]["elapsed_ns"]

        print("\nTASK 4 — SUMMARY")
        print(f"Stride 1 / 8 slowdown:  {t1 / t8:.4f}x")
        print(f"Stride 1 / 16 slowdown: {t1 / t16:.4f}x")
        print("TASK 4 PASSED", flush=True)

    return results if rank == 0 else None
