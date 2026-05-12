# CUDA Sparse DR-BCG

This project implements a sparse GPU version of block DR-BCG for symmetric positive definite (SPD) linear systems with multiple right-hand sides.

## How to run

Build from the project root:

```bash
NVHPC=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3
CUDA_VER=12.8

nvcc -O2 -std=c++17 dr_bcg_cuda_sparse.cu \
  -L$NVHPC/math_libs/$CUDA_VER/lib64 \
  -L$NVHPC/cuda/$CUDA_VER/lib64 \
  -lcublas -lcusolver -lcusparse -lz -o bcg
```

Run on one of the provided MATLAB sparse matrices:

```bash
./bcg data/apache1.mat
```

The program loads the matrix, builds a synthetic block right-hand side, runs both CPU and GPU DR-BCG, and prints timing, residual, and speedup information.

## Required packages / dependencies

- `nvcc` with C++17 support
- CUDA runtime
- cuBLAS
- cuSPARSE
- cuSOLVER
- zlib

## Required hardware / software environment

- NVIDIA GPU
- Linux x86_64 environment with CUDA libraries available at build and run time
- Example build settings in the source use NVIDIA HPC SDK `25.3` and CUDA `12.8`

## Main code locations

- Main GPU solver: `dr_bcg_algorithm5(...)` in `dr_bcg_cuda_sparse.cu`, starts at line `1003`
- Benchmark / demo driver: `main(...)` in `dr_bcg_cuda_sparse.cu`, starts at line `1143`
- Custom CUDA kernels used during QR post-processing:
  `extract_upper_triangle_kernel(...)` in `dr_bcg_cuda_sparse.cu`, starts at line `843`
  `fix_qr_signs_kernel(...)` in `dr_bcg_cuda_sparse.cu`, starts at line `855`
- Sample input matrices: `data/`
