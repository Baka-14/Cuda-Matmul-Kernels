# Matrix Multiplication CUDA Kernels

CUDA implementations of matrix multiplication: naive and tiled (shared-memory) kernels.

## Project structure

```
├── src/                    # CUDA source files
│   ├── matmul_naive.cu     # Naive kernel (one element per thread)
│   └── matmul_tiled.cu     # Tiled kernel (shared memory)
├── profiling/              # Profiling exports (Nsight Compute)
│   ├── reports/            # PDF summaries
│   │   ├── Naive Profiling Details.pdf
│   │   └── Tiled Profiling Details.pdf
│   ├── raw/                # CSV raw dumps
│   │   ├── Naive Profiling Raw dump.csv
│   │   └── Tiled Profiling Raw dump.csv
│   ├── sessions/           # .ncu-rep session files
│   │   ├── profile_naive.ncu-rep
│   │   ├── profile_tiled.ncu-rep
│   │   └── profile_unrolled.ncu-rep
│   └── README.md
└── README.md
```

## Building

From the project root, compile from `src/`:

```bash
nvcc -o matmul_naive src/matmul_naive.cu
nvcc -o matmul_tiled src/matmul_tiled.cu
```

## Kernels

- **Naive**: Each thread computes one element of C; reads from global memory.
- **Tiled**: Uses shared memory tiles (16×16) for better memory reuse and performance.
