---
layout: post
title: "A Comprehensive Summary of Multi-GPU Jacobi Solver — MPI, NCCL, and NVSHMEM"
date: 2025-11-14 10:00:00 +0900
categories: [GPU, HPC, MPI, NCCL, NVSHMEM]
---

This post provides a comprehensive overview of how a **2D Jacobi solver** can be scaled from a single GPU to multiple GPUs.  
It covers domain decomposition, halo exchange, and three major communication frameworks:

- **MPI** – traditional two-sided communication  
- **NCCL** – GPU-centric, stream-based collective communication  
- **NVSHMEM** – PGAS-style one-sided communication with in-kernel operations  

All diagrams shown below illustrate key concepts such as symmetric heaps, GPU-initiated communication, stencil updates, domain decomposition strategies, and GPUDirect.

---

# 1. Baseline Application: 2D Jacobi Solver

Jacobi iteration is a classical 2D stencil computation where each point is updated using four neighbors.

## 1.1 Single-GPU Stencil Update

<p align="center">
  <img src="/assets/img/251114_fig1_singleGPU_jacob.png" width="80%">
  <br>
  <em>Figure 1. Single-GPU Jacobi stencil update: each cell is updated from its four neighbors.</em>
</p>

The implementation consists of two nested loops `(iy, ix)` performing neighbor-based updates.

---

## 1.2 Multi-GPU Domain Decomposition

When scaling across multiple GPUs, the domain must be partitioned.

<p align="center">
  <img src="/assets/img/251114_fig2_domain_decomposisiton.png" width="85%">
  <br>
  <em>Figure 2. Horizontal, vertical, and tiled domain decomposition strategies for Jacobi.</em>
</p>

- **Horizontal stripes** minimize number of neighbors → good for latency-bound communication  
- **Vertical stripes** align well with column-major data  
- **Tiled** partitions minimize surface-to-volume ratio → optimal for bandwidth-bound communication  

---

# 2. The Need for Halo Exchange

Boundary data must be exchanged between neighboring subdomains after every iteration.

<p align="center">
  <img src="/assets/img/251114_fig3_example_halo_exchange.png" width="85%">
  <br>
  <em>Figure 3. Jacobi halo exchange: top/bottom/left/right boundary rows must be communicated.</em>
</p>

This halo exchange becomes the main communication workload in multi-GPU Jacobi solvers.

---

# 3. MPI — Message Passing Interface

MPI is the foundational model for distributed HPC applications.

## 3.1 Basic Workflow
- Initialize (`MPI_Init`)  
- Determine rank and number of processes  
- Exchange halo rows via `MPI_Sendrecv`  
- Finalize (`MPI_Finalize`)  

MPI typically uses *one process per GPU*.

---

## 3.2 CUDA-Aware MPI and GPUDirect

CUDA-aware MPI can detect GPU memory pointers automatically via UVA.  
It then enables GPUDirect technologies:

<p align="center">
  <img src="/assets/img/251114_fig4_nvidia_gpudirect.png" width="85%">
  <br>
  <em>Figure 4. GPUDirect P2P and GPUDirect RDMA: direct GPU-to-GPU transfers across NVLink, PCIe, or InfiniBand.</em>
</p>

- **GPUDirect P2P** for intra-node GPU communication  
- **GPUDirect RDMA** lets NICs directly access GPU memory across nodes  

---

## 3.3 Communication–Computation Overlap

To hide communication costs:
1. Compute boundary regions first  
2. Launch `MPI_Sendrecv` immediately  
3. Compute interior region while communication is in progress  

This is a standard optimization pattern.

---

# 4. NCCL — NVIDIA Collective Communication Library

NCCL reorganizes GPU communication around CUDA streams.

## 4.1 Key Advantages
- Communication is executed by **GPU kernels**  
- Operations are scheduled on **CUDA streams**  
- Minimizes CPU-GPU synchronization  
- Excellent overlap of communication and computation  

---

## 4.2 NCCL Halo Exchange

Halo exchange becomes GPU-driven:

- `ncclSend` / `ncclRecv` replace `MPI_Sendrecv`  
- No need for CPU synchronization  
- High-priority stream handles boundary communication  
- Low-priority stream handles interior stencil updates  

This improves concurrency and reduces latency.

---

# 5. NVSHMEM — One-Sided PGAS Communication

NVSHMEM provides a radically different communication model based on **Partitioned Global Address Space (PGAS)**.

## 5.1 Symmetric Heap

<p align="center">
  <img src="/assets/img/251114_fig5_nvshmem.png" width="85%">
  <br>
  <em>Figure 5. NVSHMEM symmetric heap: identical virtual addresses across PEs for remote memory access.</em>
</p>

Memory allocated with `nvshmem_malloc` is available at the *same* virtual address across all GPUs (PEs).  
This enables straightforward pointer-based remote access.

---

## 5.2 GPU-Initiated Communication

NVSHMEM allows GPU kernels themselves to perform communication:

<p align="center">
  <img src="/assets/img/251114_fig6_nvshmem.png" width="85%">
  <br>
  <em>Figure 6. GPU-initiated communication with NVSHMEM: eliminates CPU synchronization and launches nvshmem_put directly from inside the kernel.</em>
</p>

### Benefits
- Eliminates CPU offload latency  
- Perfect overlap of computation and communication  
- Thread-level latency hiding  
- Inline communication logic inside GPU kernels  

---

## 5.3 Two Communication Styles

### **(1) Thread-level put/get**
Each thread performs:

```cpp
nvshmem_float_p(&remote[i], local_val, pe);
```

### **(2) Block-cooperative bulk transfers**
Thread blocks cooperatively send bulk halo data using:

```cpp
nvshmem_float_put_nbi_block(...);
```

---

# 6. Fully Fused NVSHMEM Jacobi Kernel

The most advanced NVSHMEM optimization puts *the entire Jacobi iteration loop* into one GPU kernel.

Inside the kernel:
- All iterations run locally  
- Halo exchange uses `nvshmem_put`  
- Synchronization is performed using remote signals + local waits  
- No CPU involvement  
- Zero repeated kernel launch overhead  

This yields:
- Maximum overlap  
- Lowest latency  
- Best scalability  

---

# 7. Summary Table

| Framework | Communication Model | Key Characteristics | Application to Jacobi |
|----------|----------------------|----------------------|------------------------|
| **MPI** | Two-sided (send + recv) | Standard HPC, CPU-driven | Halo exchange via `MPI_Sendrecv` |
| **NCCL** | Two-sided but GPU-driven | Stream-based, optimized for NVLink/NVSwitch | Overlap via high/low priority streams |
| **NVSHMEM** | One-sided PGAS | In-kernel puts/gets, symmetric heap | Fully fused kernel, highest scalability |

---

If you want, I can also generate:  
✔ polished SVG diagrams,  
✔ step-by-step pseudocode for all three implementations,  
✔ or split this post into a multi-part blog series.
