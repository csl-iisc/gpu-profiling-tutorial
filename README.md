# CUDA Profiling Lightning-Tutorial 📊🔥

Welcome! This repo contains the **30-minute Nsight Systems (`nsys`) + Nsight Compute (`ncu`) tutorial** you can run on any CUDA-capable Linux box (or W SL / Docker image that has the CUDA Toolkit ≥ 11.0 installed).

---

## 1 . What’s Inside 🗂️

| File / Target                           | Purpose                                                                                                            |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **`gemm.cu`** *(alias `naive_gemm.cu`)* | Baseline matrix-multiply (global memory only).                                                                     |
| **`nvtx_gemm.cu`**                      | Same math + **single NVTX range** around the kernel.                                                               |
| **`pinned_gemm.cu`**                    | Adds **pinned host memory** (overlap).                                                  |
| **`tiled_gemm.cu`**                     | Shared-memory **tiled kernel** with tunable `#define BLOCK 16/32`.                                                 |
| **`Makefile`**                          | `make all` or `make naive/nvtx/pinned/tiled` builds individual demos.                                              |
| **`run.sh`**                            | One-click script that: <br>1. profiles each variant with `nsys` & `ncu` <br>2. drops reports next to the binaries. |
| `*.ncu-rep`                             | Pre-generated Nsight Compute reports you can open in the GUI if you have no GPU handy.                             |
| `*.nsys-rep`                            | Pre-generated Nsight Systems timelines (same idea).                                                                |

> **Tip:** If you’re on a headless cluster, copy the `*.rep` files to your laptop and open them there.

---

## 2 . Prerequisites ⚙️

| Need                     | Version             | Check            |
| ------------------------ | ------------------- | ---------------- |
| CUDA Toolkit             | **≥ 11.0**          | `nvcc --version` |
| Nsight Systems CLI & GUI | **2022.2** or later | `nsys --version` |
| Nsight Compute CLI & GUI | **2022.2** or later | `ncu --version`  |
| GPU driver               | Matching Toolkit    | `nvidia-smi`     |

*Inside Docker?* Use NVIDIA’s `nvcr.io/nvidia/nsight-sys-ncu:<tag>` container.

---

## 3 . Build 🛠️

```bash
git clone <this-repo>.git
cd cuda_profiling_tutorial
make            # builds all four binaries
# or individually:
make naive      # -> ./naive_gemm
make nvtx       # -> ./nvtx_gemm
make pinned     # -> ./pinned_gemm
make tiled      # -> ./tiled_gemm
```

---

## 4 . Quick Start 🚀

### 4.1 Run the baseline kernel

```bash
./naive_gemm 2048          # C = A×B (row-major), N=2048
```

### 4.2 Profile with **Nsight Systems**

```bash
nsys profile --trace=cuda -o gemm_full ./naive_gemm 2048
# Produces gemm_full.nsys-rep (timeline incl. copies + kernel)
```

Want to capture **only** the kernel (NVTX range)?

```bash
nsys profile --trace=cuda,nvtx --capture-range=nvtx \
             -o nvtx_only ./nvtx_gemm 2048
```

### 4.3 Deep dive with **Nsight Compute**

```bash
ncu --set full -o tiled_report ./tiled_gemm 2048
# Creates tiled_report.ncu-rep  (open in GUI)
```

Text-dump the raw page:

```bash
ncu --import tiled_report.ncu-rep --page raw --csv \
    > tiled_raw.csv
```

---

## 5 . What to Demo (cheat-sheet)

| Variant           | Focus                                | Command snippet                                         |
| ----------------- | ------------------------------------ | ------------------------------------------------------- |
| **`naive_gemm`**  | *Timeline anatomy* (copies ≫ kernel) | `nsys profile --trace=cuda …`                           |
| **`nvtx_gemm`**   | NVTX + `--capture-range`             | `nsys profile --trace=cuda,nvtx --capture-range=nvtx …` |
| **`pinned_gemm`** | Overlap copies with kernel           | Compare two `.nsys-rep` files                           |
| **`tiled_gemm`**  | SM occupancy / Roofline              | `ncu --metrics sm__throughput… ./tiled_gemm`            |

---

## 6 . Cleanup 🧹

```bash
make clean          # remove binaries
rm -f *.nsys-rep *.ncu-rep
```

---

