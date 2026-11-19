import os
import sys
import json
import math
import hashlib
import subprocess
import numpy as np

from mpi4py import MPI
from numba import cuda, float64

TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2
GLOBAL_NX = 512
GLOBAL_NY = 512
STEPS = 400
LAMBDA_DAMP = 0.15


def get_gpu_uuid() -> str:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid",
         "--format=csv,noheader"],
        encoding="utf-8"
    )
    return output.strip().splitlines()[0].strip()


def derive_student_seed(github_user: str, gpu_uuid: str) -> int:
    digest = hashlib.sha256(
        f"{github_user.strip()}::{gpu_uuid}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16)


# ============================================================
# TASK 1: Distributed Cartesian Topology & Hardware Context
# ============================================================

def init_distributed_context(github_user: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 4:
        raise RuntimeError(
            f"Expected exactly 4 MPI ranks, got {size}"
        )

    cuda.select_device(0)

    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = comm.bcast(gpu_uuid, root=0)

    instance_seed = derive_student_seed(
        github_user, gpu_uuid
    )

    cart_comm = comm.Create_cart(
        dims=[2, 2],
        periods=[False, False],
        reorder=False
    )

    coords = cart_comm.Get_coords(rank)

    nbr_up, nbr_down = cart_comm.Shift(0, 1)
    nbr_left, nbr_right = cart_comm.Shift(1, 1)

    return {
        "comm": comm,
        "cart_comm": cart_comm,
        "rank": rank,
        "size": size,
        "coords": coords,
        "neighbors": (
            nbr_up, nbr_down,
            nbr_left, nbr_right
        ),
        "instance_seed": instance_seed,
        "gpu_uuid": gpu_uuid
    }


# ============================================================
# TASK 2: Seeded Heterogeneous Subdomain Initialization
# ============================================================

def initialize_subdomains(ctx):
    row_rank, col_rank = ctx["coords"]
    nbr_up, nbr_down, nbr_left, nbr_right = ctx["neighbors"]

    nx = GLOBAL_NX // 2
    ny = GLOBAL_NY // 2

    dx = 1.0 / (GLOBAL_NX + 1)
    dy = 1.0 / (GLOBAL_NY + 1)

    # Local arrays: 256x256 owned cells + ghost halo.
    shape = (ny + 2, nx + 2)

    # Global coordinates for local indices 0..257.
    # Local index 1 corresponds to the first owned cell.
    global_i = row_rank * ny + np.arange(ny + 2)
    global_j = col_rank * nx + np.arange(nx + 2)

    y = global_i[:, None] * dy
    x = global_j[None, :] * dx

    # Heterogeneous diffusivity field.
    # The mathematical range is [0.08, 0.24].
    kappa = (
        0.16
        + 0.08
        * np.sin(2.0 * np.pi * x)
        * np.cos(2.0 * np.pi * y)
    )
    kappa = np.broadcast_to(kappa, shape).copy()

    # Deterministic Gaussian pulses derived from INSTANCE_SEED.
    rng = np.random.default_rng(ctx["instance_seed"])

    u = np.zeros(shape, dtype=np.float64)

    for _ in range(3):
        cx = rng.uniform(0.2, 0.8)
        cy = rng.uniform(0.2, 0.8)
        amplitude = rng.uniform(0.5, 1.5)
        sigma = rng.uniform(0.035, 0.075)

        u += amplitude * np.exp(
            -((x - cx)**2 + (y - cy)**2)
            / (2.0 * sigma**2)
        )

    # Localized heat source.
    source = 0.05 * np.exp(
        -((x - 0.5)**2 + (y - 0.5)**2)
        / (2.0 * 0.06**2)
    )
    source = np.broadcast_to(source, shape).copy()

    # Fixed zero-temperature physical boundaries.
    if nbr_up == MPI.PROC_NULL:
        u[0, :] = 0.0

    if nbr_down == MPI.PROC_NULL:
        u[-1, :] = 0.0

    if nbr_left == MPI.PROC_NULL:
        u[:, 0] = 0.0

    if nbr_right == MPI.PROC_NULL:
        u[:, -1] = 0.0

    return {
        "u": cuda.to_device(u),
        "kappa": cuda.to_device(kappa),
        "source": cuda.to_device(source),
        "nx": nx,
        "ny": ny,
        "dx": dx,
        "dy": dy
    }




# ============================================================
# TASK 3: GPU Halo Pack/Unpack & Pinned MPI Exchange
# ============================================================

# Direction indices
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# Tags identify the direction in which data travels.
SEND_TAGS = (10, 11, 12, 13)
OPPOSITE = (DOWN, UP, RIGHT, LEFT)


@cuda.jit
def pack_halos_kernel(u, up, down, left, right, ny, nx):
    i = cuda.grid(1)

    if i < nx:
        up[i] = u[1, i + 1]
        down[i] = u[ny, i + 1]

    if i < ny:
        left[i] = u[i + 1, 1]
        right[i] = u[i + 1, nx]


@cuda.jit
def unpack_halos_kernel(u, up, down, left, right, ny, nx):
    i = cuda.grid(1)

    if i < nx:
        u[0, i + 1] = up[i]
        u[ny + 1, i + 1] = down[i]

    if i < ny:
        u[i + 1, 0] = left[i]
        u[i + 1, nx + 1] = right[i]


def create_halo_buffers(ny, nx):
    lengths = (nx, nx, ny, ny)

    device_send = [
        cuda.device_array(n, dtype=np.float64)
        for n in lengths
    ]
    device_recv = [
        cuda.device_array(n, dtype=np.float64)
        for n in lengths
    ]

    host_send = [
        cuda.pinned_array(n, dtype=np.float64)
        for n in lengths
    ]
    host_recv = [
        cuda.pinned_array(n, dtype=np.float64)
        for n in lengths
    ]

    return {
        "device_send": device_send,
        "device_recv": device_recv,
        "host_send": host_send,
        "host_recv": host_recv
    }


def exchange_halos_async(ctx, u, buffers, stream_boundary):
    cart = ctx["cart_comm"]
    neighbors = ctx["neighbors"]

    ny = u.shape[0] - 2
    nx = u.shape[1] - 2

    ds = buffers["device_send"]
    dr = buffers["device_recv"]
    hs = buffers["host_send"]
    hr = buffers["host_recv"]

    threads = 256
    blocks = (max(nx, ny) + threads - 1) // threads

    # A: Pack borders on the GPU.
    pack_halos_kernel[blocks, threads, stream_boundary](
        u, ds[UP], ds[DOWN], ds[LEFT], ds[RIGHT], ny, nx
    )

    # Asynchronous device-to-host transfers.
    for d in range(4):
        if neighbors[d] != MPI.PROC_NULL:
            ds[d].copy_to_host(hs[d], stream=stream_boundary)

    # MPI must not read buffers until DMA is complete.
    stream_boundary.synchronize()

    # B: Non-blocking MPI exchange.
    requests = []

    for d in range(4):
        neighbor = neighbors[d]
        if neighbor == MPI.PROC_NULL:
            hr[d].fill(0.0)
            continue

        requests.append(
            cart.Irecv(
                hr[d],
                source=neighbor,
                tag=SEND_TAGS[OPPOSITE[d]]
            )
        )
        requests.append(
            cart.Isend(
                hs[d],
                dest=neighbor,
                tag=SEND_TAGS[d]
            )
        )

    # Return outstanding requests so Task 4 can
    # compute the interior while MPI transfers run.
    return requests


def finish_halo_exchange(
    ctx, u, buffers, stream_boundary, requests
):
    MPI.Request.Waitall(requests)

    dr = buffers["device_recv"]
    hr = buffers["host_recv"]

    ny = u.shape[0] - 2
    nx = u.shape[1] - 2

    for d in range(4):
        dr[d].copy_to_device(hr[d], stream=stream_boundary)

    threads = 256
    blocks = (max(nx, ny) + threads - 1) // threads

    unpack_halos_kernel[blocks, threads, stream_boundary](
        u, dr[UP], dr[DOWN], dr[LEFT], dr[RIGHT], ny, nx
    )

    stream_boundary.synchronize()



def validate_task3(ctx, fields):
    cart = ctx["cart_comm"]
    rank = ctx["rank"]
    neighbors = ctx["neighbors"]

    ny = fields["ny"]
    nx = fields["nx"]

    # Use a dedicated test array so Task 2 data
    # remains unchanged.
    test_host = np.full(
        (ny + 2, nx + 2), -999.0, dtype=np.float64
    )

    # Each rank has a unique interior value.
    test_host[1:-1, 1:-1] = float(rank + 1)

    test_device = cuda.to_device(test_host)

    buffers = create_halo_buffers(ny, nx)
    stream_boundary = cuda.stream()

    requests = exchange_halos_async(
        ctx, test_device, buffers, stream_boundary
    )

    finish_halo_exchange(
        ctx, test_device, buffers,
        stream_boundary, requests
    )

    result = test_device.copy_to_host()

    up, down, left, right = neighbors

    expected = [
        (up, result[0, 1:-1]),
        (down, result[-1, 1:-1]),
        (left, result[1:-1, 0]),
        (right, result[1:-1, -1])
    ]

    for neighbor, halo in expected:
        value = (
            0.0 if neighbor == MPI.PROC_NULL
            else float(neighbor + 1)
        )
        assert np.all(halo == value), (
            f"Rank {rank}: halo mismatch "
            f"for neighbor {neighbor}"
        )

    print(
        f"Rank {rank} | "
        f"GPU halo pack/unpack=PASS | "
        f"Pinned MPI exchange=PASS",
        flush=True
    )

    cart.Barrier()

    if rank == 0:
        print("TASK 3 VALIDATION PASSED", flush=True)



# ============================================================
# TASK 4: SHARED-MEMORY TILED STENCIL & OVERLAP
# ============================================================

@cuda.jit
def tiled_stencil_kernel(
    u, u_next, kappa, source,
    dx, dy, dt, damping, region
):
    tile = cuda.shared.array(
        shape=(18, 18),
        dtype=float64
    )

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    x = cuda.blockIdx.x * 16 + tx + 1
    y = cuda.blockIdx.y * 16 + ty + 1

    nx = u.shape[1] - 2
    ny = u.shape[0] - 2

    sx = tx + 1
    sy = ty + 1

    # Load center and halo values into shared memory.
    # Every thread participates in synchronization.
    if x <= nx and y <= ny:
        tile[sy, sx] = u[y, x]

    if tx == 0 and y <= ny:
        tile[sy, 0] = u[y, x - 1]

    if tx == 15 and y <= ny and x <= nx:
        tile[sy, 17] = u[y, x + 1]

    if ty == 0 and x <= nx:
        tile[0, sx] = u[y - 1, x]

    if ty == 15 and x <= nx and y <= ny:
        tile[17, sx] = u[y + 1, x]

    cuda.syncthreads()

    if x > nx or y > ny:
        return

    is_interior = (
        1 < x < nx and
        1 < y < ny
    )

    if region == 0 and not is_interior:
        return

    if region == 1 and is_interior:
        return

    # Read neighboring values. Global-memory fallback
    # handles partial tiles at the domain edges.
    center = tile[sy, sx]

    west = u[y, x - 1]
    east = u[y, x + 1]
    north = u[y - 1, x]
    south = u[y + 1, x]

    if tx > 0:
        west = tile[sy, sx - 1]

    if tx < 15 and x < nx:
        east = tile[sy, sx + 1]

    if ty > 0:
        north = tile[sy - 1, sx]

    if ty < 15 and y < ny:
        south = tile[sy + 1, sx]

    laplacian = (
        (west - 2.0 * center + east) / (dx * dx)
        +
        (north - 2.0 * center + south) / (dy * dy)
    )

    u_next[y, x] = (
        center
        + dt * (
            kappa[y, x] * laplacian
            + source[y, x]
            - damping * center
        )
    )


def step_overlapped(
    ctx, fields, buffers,
    stream_compute, stream_boundary, dt
):
    u = fields["u"]
    u_next = fields["u_next"]

    nx = fields["nx"]
    ny = fields["ny"]

    threads = (16, 16)
    blocks = (
        (nx + 15) // 16,
        (ny + 15) // 16
    )

    # Start nonblocking MPI communication.
    requests = exchange_halos_async(
        ctx, u, buffers, stream_boundary
    )

    # Compute interior while MPI requests are active.
    tiled_stencil_kernel[
        blocks, threads, stream_compute
    ](
        u, u_next,
        fields["kappa"], fields["source"],
        fields["dx"], fields["dy"],
        dt, LAMBDA_DAMP, 0
    )

    # Wait for communication and update ghost cells.
    finish_halo_exchange(
        ctx, u, buffers,
        stream_boundary, requests
    )

    # Compute cells adjacent to subdomain boundaries.
    tiled_stencil_kernel[
        blocks, threads, stream_boundary
    ](
        u, u_next,
        fields["kappa"], fields["source"],
        fields["dx"], fields["dy"],
        dt, LAMBDA_DAMP, 1
    )

    stream_compute.synchronize()
    stream_boundary.synchronize()

    fields["u"], fields["u_next"] = (
        fields["u_next"], fields["u"]
    )




# ============================================================
# TASK 4 VALIDATION
# ============================================================

def validate_task4(ctx, fields):
    nx = fields["nx"]
    ny = fields["ny"]

    # Preserve the original Task 2 data
    test_fields = dict(fields)

    test_fields["u"] = cuda.to_device(
        fields["u"].copy_to_host()
    )

    test_fields["u_next"] = cuda.device_array_like(
        test_fields["u"]
    )

    buffers = create_halo_buffers(ny, nx)

    stream_compute = cuda.stream()
    stream_boundary = cuda.stream()

    dx = fields["dx"]
    dy = fields["dy"]

    max_kappa = 0.24

    dt = 0.20 / (
        max_kappa * (
            1.0 / dx**2 + 1.0 / dy**2
        )
    )

    before = test_fields["u"].copy_to_host()

    step_overlapped(
        ctx, test_fields, buffers,
        stream_compute, stream_boundary, dt
    )

    after = test_fields["u"].copy_to_host()

    assert after.shape == before.shape
    assert np.isfinite(after[1:-1, 1:-1]).all()

    change = np.max(
        np.abs(
            after[1:-1, 1:-1]
            - before[1:-1, 1:-1]
        )
    )

    assert change > 0.0, (
        "Stencil did not update the field"
    )

    print(
        f"Rank {ctx['rank']} | "
        f"Stencil=PASS | "
        f"max_change={change:.8f}",
        flush=True
    )

    ctx["cart_comm"].Barrier()

    if ctx["rank"] == 0:
        print(
            "TASK 4 VALIDATION PASSED",
            flush=True
        )


# ============================================================
# TASK 4: SEQUENTIAL BASELINE FOR BENCHMARK
# ============================================================

def step_sequential(
    ctx, fields, buffers,
    stream_compute, stream_boundary, dt
):
    u = fields["u"]
    u_next = fields["u_next"]

    nx = fields["nx"]
    ny = fields["ny"]

    threads = (16, 16)
    blocks = (
        (nx + 15) // 16,
        (ny + 15) // 16
    )

    # Complete MPI communication first
    requests = exchange_halos_async(
        ctx, u, buffers, stream_boundary
    )

    finish_halo_exchange(
        ctx, u, buffers,
        stream_boundary, requests
    )

    # Compute all cells after communication
    tiled_stencil_kernel[
        blocks, threads, stream_compute
    ](
        u, u_next,
        fields["kappa"], fields["source"],
        fields["dx"], fields["dy"],
        dt, LAMBDA_DAMP, 2
    )

    stream_compute.synchronize()
    stream_boundary.synchronize()

    fields["u"], fields["u_next"] = (
        fields["u_next"], fields["u"]
    )

# ============================================================
# TASK 5: DISTRIBUTED SOLVER & BENCHMARK
# ============================================================

def run_solver(ctx, steps=STEPS):
    comm = ctx["cart_comm"]
    rank = ctx["rank"]

    fields = initialize_subdomains(ctx)

    nx = fields["nx"]
    ny = fields["ny"]

    fields["u_next"] = cuda.device_array_like(
        fields["u"]
    )

    buffers = create_halo_buffers(ny, nx)

    stream_compute = cuda.stream()
    stream_boundary = cuda.stream()

    dx = fields["dx"]
    dy = fields["dy"]

    # Global maximum diffusivity for stable time step.
    local_kappa_max = float(
        np.max(fields["kappa"].copy_to_host())
    )

    global_kappa_max = comm.allreduce(
        local_kappa_max, op=MPI.MAX
    )

    dt = 0.20 / (
        global_kappa_max *
        (1.0 / dx**2 + 1.0 / dy**2)
    )

    # Warm up CUDA JIT compilation before timing.
    # Use a separate initialized state so the
    # measured simulation still starts at step zero.
    warmup_fields = dict(fields)
    warmup_fields["u"] = cuda.to_device(
        fields["u"].copy_to_host()
    )
    warmup_fields["u_next"] = cuda.device_array_like(
        warmup_fields["u"]
    )

    step_overlapped(
        ctx, warmup_fields, buffers,
        stream_compute, stream_boundary, dt
    )

    cuda.synchronize()
    comm.Barrier()

    start = MPI.Wtime()

    for step in range(steps):
        step_overlapped(
            ctx, fields, buffers,
            stream_compute, stream_boundary, dt
        )

    cuda.synchronize()
    comm.Barrier()

    elapsed_local = MPI.Wtime() - start

    elapsed = comm.allreduce(
        elapsed_local, op=MPI.MAX
    )

    result = fields["u"].copy_to_host()
    interior = result[1:-1, 1:-1]

    local_valid = bool(np.isfinite(interior).all())

    all_valid = comm.allreduce(
        local_valid, op=MPI.LAND
    )

    if not all_valid:
        raise RuntimeError(
            "Non-finite values detected in solver"
        )

    local_sum = float(np.sum(interior))
    local_min = float(np.min(interior))
    local_max = float(np.max(interior))

    global_sum = comm.allreduce(
        local_sum, op=MPI.SUM
    )
    global_min = comm.allreduce(
        local_min, op=MPI.MIN
    )
    global_max = comm.allreduce(
        local_max, op=MPI.MAX
    )

    global_count = comm.allreduce(
        interior.size, op=MPI.SUM
    )

    global_mean = global_sum / global_count

    metrics = {
        "github_username": None,
        "gpu_uuid": ctx["gpu_uuid"],
        "instance_seed": ctx["instance_seed"],
        "mpi_ranks": ctx["size"],
        "global_nx": GLOBAL_NX,
        "global_ny": GLOBAL_NY,
        "steps": steps,
        "dt": dt,
        "runtime_seconds": elapsed,
        "milliseconds_per_step": elapsed * 1000 / steps,
        "global_min": global_min,
        "global_max": global_max,
        "global_mean": global_mean,
        "global_sum": global_sum
    }

    if rank == 0:
        print("=" * 55)
        print("TASK 5 — DISTRIBUTED CUDA SOLVER")
        print("=" * 55)
        print(f"MPI processes: {ctx['size']}")
        print(f"Grid: {GLOBAL_NX} x {GLOBAL_NY}")
        print(f"Steps: {steps}")
        print(f"Instance seed: {ctx['instance_seed']}")
        print(f"dt: {dt:.12e}")
        print(f"Runtime: {elapsed:.6f} seconds")
        print(
            f"Time per step: "
            f"{elapsed * 1000 / steps:.4f} ms"
        )
        print(f"Global min: {global_min:.8f}")
        print(f"Global max: {global_max:.8f}")
        print(f"Global mean: {global_mean:.8f}")
        print("=" * 55)
        print("TASK 5 SOLVER COMPLETED", flush=True)

    return metrics


# ============================================================
# TASK 4: CPU REFERENCE ACCURACY VALIDATION
# ============================================================

def validate_task4_accuracy(ctx, fields):
    rank = ctx["rank"]
    nx = fields["nx"]
    ny = fields["ny"]

    # Independent test copy
    test_fields = dict(fields)
    test_fields["u"] = cuda.to_device(
        fields["u"].copy_to_host()
    )
    test_fields["u_next"] = cuda.device_array_like(
        test_fields["u"]
    )

    buffers = create_halo_buffers(ny, nx)
    stream_compute = cuda.stream()
    stream_boundary = cuda.stream()

    dx = fields["dx"]
    dy = fields["dy"]

    dt = 0.20 / (
        0.24 * (1.0 / dx**2 + 1.0 / dy**2)
    )

    # Run one distributed CUDA step
    step_overlapped(
        ctx, test_fields, buffers,
        stream_compute, stream_boundary, dt
    )

    gpu_result = (
        test_fields["u"].copy_to_host()
    )

    # The exchanged ghost cells are now in the
    # previous u array after buffer swapping.
    u_before = (
        test_fields["u_next"].copy_to_host()
    )

    kappa = fields["kappa"].copy_to_host()
    source = fields["source"].copy_to_host()

    center = u_before[1:-1, 1:-1]

    laplacian = (
        (
            u_before[1:-1, :-2]
            - 2.0 * center
            + u_before[1:-1, 2:]
        ) / (dx * dx)
        +
        (
            u_before[:-2, 1:-1]
            - 2.0 * center
            + u_before[2:, 1:-1]
        ) / (dy * dy)
    )

    cpu_result = (
        center
        + dt * (
            kappa[1:-1, 1:-1] * laplacian
            + source[1:-1, 1:-1]
            - LAMBDA_DAMP * center
        )
    )

    gpu_interior = gpu_result[1:-1, 1:-1]

    max_error = np.max(
        np.abs(gpu_interior - cpu_result)
    )

    assert np.isfinite(max_error)
    assert max_error < 1e-10, (
        f"Rank {rank}: CPU/GPU mismatch "
        f"(max_error={max_error})"
    )

    print(
        f"Rank {rank} | "
        f"CPU/GPU accuracy=PASS | "
        f"max_error={max_error:.3e}",
        flush=True
    )

    ctx["cart_comm"].Barrier()

    if rank == 0:
        print(
            "TASK 4 ACCURACY VALIDATION PASSED",
            flush=True
        )




# ============================================================
# TASK 5: SEQUENTIAL VS OVERLAPPED BENCHMARK
# ============================================================

def benchmark_solver(ctx, steps=STEPS):
    comm = ctx["cart_comm"]
    rank = ctx["rank"]

    results = {}

    for mode in ("sequential", "overlapped"):
        # Fresh initial conditions for each mode
        fields = initialize_subdomains(ctx)

        nx = fields["nx"]
        ny = fields["ny"]

        fields["u_next"] = cuda.device_array_like(
            fields["u"]
        )

        buffers = create_halo_buffers(ny, nx)

        stream_compute = cuda.stream()
        stream_boundary = cuda.stream()

        dx = fields["dx"]
        dy = fields["dy"]

        local_kappa_max = float(
            np.max(fields["kappa"].copy_to_host())
        )

        global_kappa_max = comm.allreduce(
            local_kappa_max, op=MPI.MAX
        )

        dt = 0.20 / (
            global_kappa_max *
            (1.0 / dx**2 + 1.0 / dy**2)
        )

        step_fn = (
            step_sequential
            if mode == "sequential"
            else step_overlapped
        )

        # Warm up kernels without changing initial state
        warmup = dict(fields)
        warmup["u"] = cuda.to_device(
            fields["u"].copy_to_host()
        )
        warmup["u_next"] = cuda.device_array_like(
            warmup["u"]
        )

        step_fn(
            ctx, warmup, buffers,
            stream_compute, stream_boundary, dt
        )

        cuda.synchronize()
        comm.Barrier()

        start = MPI.Wtime()

        for _ in range(steps):
            step_fn(
                ctx, fields, buffers,
                stream_compute, stream_boundary, dt
            )

        cuda.synchronize()
        elapsed_local = MPI.Wtime() - start

        elapsed = comm.allreduce(
            elapsed_local, op=MPI.MAX
        )

        result = fields["u"].copy_to_host()
        interior = result[1:-1, 1:-1]

        local_valid = bool(np.isfinite(interior).all())
        all_valid = comm.allreduce(
            local_valid, op=MPI.LAND
        )

        if not all_valid:
            raise RuntimeError(
                f"{mode}: non-finite result"
            )

        local_sum = float(np.sum(interior))
        global_sum = comm.allreduce(
            local_sum, op=MPI.SUM
        )

        results[mode] = {
            "seconds": elapsed,
            "ms_per_step": elapsed * 1000 / steps,
            "global_sum": global_sum,
        }

        comm.Barrier()

    sequential = results["sequential"]
    overlapped = results["overlapped"]

    speedup = (
        sequential["seconds"] /
        overlapped["seconds"]
    )

    sum_difference = abs(
        sequential["global_sum"] -
        overlapped["global_sum"]
    )

    if rank == 0:
        print("=" * 58)
        print("TASK 5 — PERFORMANCE BENCHMARK")
        print("=" * 58)
        print(f"MPI ranks: {ctx['size']}")
        print(f"Steps: {steps}")
        print()
        print(
            f"Sequential: {sequential['seconds']:.6f} s"
        )
        print(
            f"Overlapped: {overlapped['seconds']:.6f} s"
        )
        print(f"Speedup: {speedup:.4f}x")
        print(
            f"Sequential ms/step: "
            f"{sequential['ms_per_step']:.4f}"
        )
        print(
            f"Overlapped ms/step: "
            f"{overlapped['ms_per_step']:.4f}"
        )
        print(
            f"Global sum difference: "
            f"{sum_difference:.12e}"
        )
        print("=" * 58)
        print("TASK 5 BENCHMARK COMPLETED", flush=True)

    return results




if __name__ == "__main__":
    github_user = sys.argv[1]

    ctx = init_distributed_context(github_user)
    fields = initialize_subdomains(ctx)

    rank = ctx["rank"]
    cart = ctx["cart_comm"]
    up, down, left, right = ctx["neighbors"]

    u = fields["u"].copy_to_host()
    kappa = fields["kappa"].copy_to_host()
    source = fields["source"].copy_to_host()

    # Shape checks
    assert u.shape == (258, 258)
    assert kappa.shape == (258, 258)
    assert source.shape == (258, 258)

    # Numerical checks
    assert np.isfinite(u).all()
    assert np.isfinite(kappa).all()
    assert np.isfinite(source).all()

    assert kappa.min() >= 0.08 - 1e-12
    assert kappa.max() <= 0.24 + 1e-12

    assert np.max(u[1:-1, 1:-1]) > 0.0
    assert np.min(u[1:-1, 1:-1]) >= 0.0

    # Physical boundary checks
    if up == MPI.PROC_NULL:
        assert np.all(u[0, :] == 0.0)

    if down == MPI.PROC_NULL:
        assert np.all(u[-1, :] == 0.0)

    if left == MPI.PROC_NULL:
        assert np.all(u[:, 0] == 0.0)

    if right == MPI.PROC_NULL:
        assert np.all(u[:, -1] == 0.0)

    # Check coordinate consistency between MPI neighbors.
    # Exchange kappa border values only for validation.
    for axis, negative, positive in [
        (0, up, down),
        (1, left, right)
    ]:
        if axis == 0:
            send_neg = kappa[1, 1:-1].copy()
            send_pos = kappa[-2, 1:-1].copy()
            expected_neg = kappa[0, 1:-1]
            expected_pos = kappa[-1, 1:-1]
        else:
            send_neg = kappa[1:-1, 1].copy()
            send_pos = kappa[1:-1, -2].copy()
            expected_neg = kappa[1:-1, 0]
            expected_pos = kappa[1:-1, -1]

        recv_neg = np.empty_like(send_neg)
        recv_pos = np.empty_like(send_pos)

        cart.Sendrecv(
            sendbuf=send_pos,
            dest=positive,
            sendtag=axis,
            recvbuf=recv_neg,
            source=negative,
            recvtag=axis
        )

        cart.Sendrecv(
            sendbuf=send_neg,
            dest=negative,
            sendtag=axis + 2,
            recvbuf=recv_pos,
            source=positive,
            recvtag=axis + 2
        )

        if negative != MPI.PROC_NULL:
            assert np.allclose(
                recv_neg, expected_neg, atol=1e-14, rtol=0
            )

        if positive != MPI.PROC_NULL:
            assert np.allclose(
                recv_pos, expected_pos, atol=1e-14, rtol=0
            )

    print(
        f"Rank {rank} | "
        f"Shape={u.shape} | "
        f"u_max={u[1:-1, 1:-1].max():.6f} | "
        f"kappa_min={kappa.min():.6f} | "
        f"kappa_max={kappa.max():.6f} | "
        f"Checks=PASS",
        flush=True
    )

    cart.Barrier()

    if rank == 0:
        print("TASK 2 VALIDATION PASSED", flush=True)

    validate_task3(ctx, fields)
    validate_task4(ctx, fields)
    validate_task4_accuracy(ctx, fields)

    # TASK 5: Full distributed simulation
    metrics = run_solver(ctx, steps=STEPS)

    if ctx["rank"] == 0:
        metrics["github_username"] = github_user

        with open(
            "solver_metrics.json", "w",
            encoding="utf-8"
        ) as f:
            json.dump(metrics, f, indent=4)

        print(
            "Metrics saved to solver_metrics.json",
            flush=True
        )

    # TASK 5: Performance comparison
    benchmark_solver(ctx, steps=STEPS)
