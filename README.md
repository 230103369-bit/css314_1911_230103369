# CSS314 — Parallel Computing Labs

**Student ID:** 230103369  
**GitHub:** 230103369-bit  
**Course:** CSS314 — Parallel Computing

This repository contains two laboratory assignments focused on parallel computing, distributed execution, GPU acceleration, and cache-coherence performance.

## Lab 1 — MPI + CUDA Stencil

This laboratory explores distributed and GPU-based parallel computing through a stencil computation.

**Files:**
- `distributed_cuda_solver.py` — Python implementation.
- `mpi_cuda_stencil_lab.ipynb` — Google Colab notebook.
- `submission_manifest.json` — Submission metadata and experimental results.

## Lab 2 — Cache Line & False Sharing Mechanics

This laboratory investigates cache-line contention and false sharing in multithreaded Python using Numba-compiled native kernels.

**Experimental configuration:**
- Processor: Apple M2 (8 logical cores)
- Cache line size: 128 bytes
- Threads: 4
- Iterations: 60,000,000 per thread
- Strides: 1, 4, 8, and 16

### Benchmark Results

| Stride | Execution Time (ms) |
|---|---:|
| 1 | 135.790 |
| 4 | 147.021 |
| 8 | 130.852 |
| 16 | 127.920 |
| Register-private | 0.096 |

All counter parity checks and the implemented LLVM audit passed.

**Files:**
- `false_sharing_lab.py` — Multithreaded benchmark implementation.
- `cache_lab_manifest.json` — System information, benchmark measurements, and verification results.
- `ANALYSIS.md` — Answers to forensic questions Q1–Q5.

## Repository Structure

```text
css314_1911_230103369/
├── distributed_cuda_solver.py
├── mpi_cuda_stencil_lab.ipynb
├── submission_manifest.json
├── false_sharing_lab.py
├── cache_lab_manifest.json
├── ANALYSIS.md
└── README.md
```

## Submission

Both laboratory assignments are maintained in this single repository as instructed by the course instructor.
