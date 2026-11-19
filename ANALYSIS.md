# Cache Line & False Sharing Mechanics in Multi-Core Python

**Student GitHub:** `230103369-bit`  
**CPU:** Apple M2 (ARM64)  
**Logical cores:** 8  
**Detected cache line size:** 128 bytes  
**Python:** 3.14.7  
**NumPy:** 2.5.4  
**Threads:** 4  
**Iterations per thread:** 60,000,000  
**Total updates per benchmark:** 240,000,000  
**Instance seed:** `4774598612729952269`

## Q1 — MESI Protocol State Transitions

False sharing occurs when two or more cores modify independent variables located within the same cache line. Although the memory addresses are different, the hardware coherence mechanism operates at cache-line granularity.

For `stride = 1`, Core 0 writes to `buffer[0]`, while Core 1 writes to `buffer[1]`. These `uint64` elements are separated by only 8 bytes and therefore occupy the same cache line.

A simplified MESI sequence is:

1. Both cores initially have an invalid copy (I) of the cache line.
2. Core 0 obtains ownership through a Read-For-Ownership (RFO) transaction and moves the line into the Modified (M) state after its store.
3. Core 1 requests ownership of the same line. The coherence protocol transfers ownership, invalidating Core 0's copy.
4. Core 1 modifies its element, placing its copy in the Modified state.
5. When Core 0 writes again, ownership must transfer back, potentially causing repeated cache-coherence traffic.

A line may also pass through Shared (S) or Exclusive (E) states depending on preceding reads and the coherence implementation. The exact hardware state sequence was not measured in this experiment.

An RFO transaction can delay a store because the requesting core must obtain exclusive ownership before its write becomes globally coherent. Repeated ownership transfers increase coherence latency even though the threads update different array elements.

The Apple M2 uses ARM64 hardware, so MESI is used here as a conceptual model rather than a claim about its exact proprietary coherence-state implementation.

## Q2 — Quantitative Slowdown and Coherency Overhead

The benchmark executed four native threads, each performing 60 million updates.

| Layout | Stride | Elapsed time (ns) | ns/update |
|---|---:|---:|---:|
| Contiguous | 1 | 135,789,666 | 0.565790 |
| Half-line (64-byte reference) | 4 | 147,020,541 | 0.612586 |
| Line-padded (64-byte reference) | 8 | 130,852,167 | 0.545217 |
| 128-byte spacing | 16 | 127,919,625 | 0.532998 |
| Register-private | 8 | 95,542 | 0.000398 |

The measured slowdown factor of stride 1 relative to stride 8 was:

**135,789,666 / 130,852,167 = 1.037733×**

Therefore, stride 1 was approximately **3.77% slower** than stride 8.

The stride 4 configuration was the slowest in this run, while stride 16 was the fastest among the shared-memory update configurations.

The register-private baseline completed in 95,542 ns. This result is much faster because it does not require a shared-memory store on every iteration. It should not be interpreted as a direct measurement of the cost of a single physical memory update, since the compiler can optimize the private accumulation loop substantially.

The measured times are wall-clock durations divided by 240 million logical updates. Because four threads execute concurrently, these values represent aggregate throughput-based averages, not individual instruction latencies.

## Q3 — L2 Spatial Prefetcher and 128-Byte Pairing

The measured results were:

- Stride 8: 130,852,167 ns
- Stride 16: 127,919,625 ns

Stride 16 was **2,932,542 ns faster**, corresponding to approximately **2.24% less elapsed time** than stride 8.

The assignment describes 64-byte cache lines and adjacent-line prefetching that may operate on 128-byte regions. However, the actual Apple M2 system reported a cache line size of **128 bytes**.

Consequently, for the measured system:

- Stride 8 corresponds to 64-byte spacing.
- Stride 16 corresponds to 128-byte spacing.

With a suitably aligned buffer, stride 8 can still place two thread counters in the same 128-byte cache line, whereas stride 16 separates them into distinct 128-byte lines.

This provides a plausible explanation for the lower measured latency of stride 16.

On systems with 64-byte cache lines, adjacent-line prefetchers may fetch neighboring lines together. Depending on the microarchitecture, this can create additional cache traffic even when individual counters occupy different cache lines. However, such prefetcher behavior and secondary invalidations were not directly measured on this Apple M2.

The difference is relatively small, and the operating system did not provide per-thread CPU affinity pinning. Therefore, scheduler placement and measurement variability may also contribute.

## Q4 — Compiler Register Promotion vs. Memory Stores

The benchmark uses a Numba native kernel compiled with `@njit(nogil=True)`.

An optimizing compiler can transform a repeated accumulation into a register-based calculation. Instead of loading and storing the same memory location on every iteration, it may load the value once, accumulate in a register, and write the result at the end.

Such an optimization would remove most of the memory traffic that the false-sharing benchmark is intended to measure.

The LLVM audit reported:

| Audit property | Recorded value |
|---|---|
| LLVM audit passed | `true` |
| Recorded i64 load count | 2 |
| Recorded i64 store count | 1 |
| Memory clobber count | 2 |
| Python object fallback | `false` |

The recorded LLVM excerpt contains:

```llvm
tail call void asm sideeffect "", "~{memory}"() #0
%.158 = load i64, ptr %sunkaddr1, align 8
store i64 %.159, ptr %sunkaddr1, align 8
tail call void asm sideeffect "", "~{memory}"() #0
%.30 = load i64, ptr %sunkaddr, align 8
```

The `load i64` and `store i64` instructions demonstrate native integer memory accesses. The inline assembly memory clobbers act as compiler barriers, restricting reordering and register promotion across those points.

The displayed memory operations are **not marked `volatile`** in the LLVM excerpt. Therefore, the evidence supports native memory access with compiler barriers, rather than LLVM volatile load/store instructions.

The manifest records `llvm_audit_passed = true`, and the kernel has no Python object fallback.

## Q5 — CPython GIL Masking vs. nogil=True Execution

In conventional CPython execution, the Global Interpreter Lock (GIL) generally prevents multiple threads from executing Python bytecode simultaneously in the same interpreter.

Consequently, a counter loop written entirely in ordinary Python does not normally generate sustained simultaneous CPU-core writes from four threads. The experiment would be dominated by interpreter overhead, GIL contention, and thread scheduling rather than cleanly isolating cache-line coherence costs.

Using `@njit(nogil=True)` allows Numba to compile the worker function into native machine code that can execute without holding the Python GIL.

The four `threading.Thread` workers can therefore perform their counter updates concurrently on available CPU cores, making hardware false sharing observable.

Increasing the number of ordinary Python threads may reduce performance because of GIL contention and context-switching overhead, while GIL-free native threads can expose cache-coherence contention.

The actual recorded context switches were:

| Layout | Voluntary | Involuntary |
|---|---:|---:|
| Stride 1 | 0 | 365 |
| Stride 4 | 0 | 358 |
| Stride 8 | 0 | 188 |
| Stride 16 | 0 | 209 |
| Register-private | 0 | 23 |

These counts were collected using process resource-usage telemetry. They do not by themselves identify cache-coherence stalls.

The Apple M2/macOS environment did not support the per-thread CPU affinity method used by the benchmark, so the workers were scheduled by the operating system.

## Verification Summary

The benchmark successfully completed all required layouts and the register-private baseline.

All four counters in every layout reached exactly **60,000,000**, demonstrating bit-exact final-state parity.

The buffers satisfied the required 64-byte alignment. The LLVM audit passed, and no Python object fallback was detected.

**Final manifest status: `verification_passed = true`**

The results show modest differences between shared-memory layouts on Apple M2, with stride 16 producing the lowest measured elapsed time among the four stride configurations. Because the processor uses 128-byte cache lines and thread pinning was unavailable, the measurements should be interpreted in the context of this specific hardware and operating-system environment.

**Source of experimental measurements:** `cache_lab_manifest.json`, generated by `false_sharing_lab.py` on the Apple M2 host.