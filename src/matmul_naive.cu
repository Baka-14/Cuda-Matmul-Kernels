#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Matrix dimensions — square matrices for simplicity
// 1024x1024 is big enough to see real GPU behavior
#define M 1024  // rows of A and C
#define K 1024  // cols of A, rows of B
#define N 1024  // cols of B and C

// Block size: 16x16 = 256 threads per block
#define BLOCK_SIZE 16


// ============================================================
// NAIVE KERNEL: Each thread computes ONE element of C
//
// Every thread reads an entire row of A and column of B
// directly from global memory (slow VRAM).
// ============================================================
__global__ void matmul_naive(float *A, float *B, float *C,
                             int m, int k, int n) {
    // Step 1: Figure out which element of C I'm responsible for
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Step 2: Bounds check — matrix might not divide evenly into blocks
    if (row < m && col < n) {
        float sum = 0.0f;

        // Step 3: Dot product — walk across row of A and column of B
        for (int i = 0; i < k; i++) {
            // A is stored row-major: A[row][i] lives at A[row * k + i]
            // B is stored row-major: B[i][col] lives at B[i * n + col]
            sum += A[row * k + i] * B[i * n + col];
        }

        // Step 4: Write result to C
        C[row * n + col] = sum;
    }
}


// ============================================================
// CPU version — to check that our GPU gives correct answers
// ============================================================
void matmul_cpu(float *A, float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// Check GPU result against CPU result
bool verify(float *gpu_result, float *cpu_result, int size) {
    for (int i = 0; i < size; i++) {
        if (fabs(gpu_result[i] - cpu_result[i]) > 1e-2) {
            printf("MISMATCH at index %d: GPU=%.6f, CPU=%.6f\n",
                   i, gpu_result[i], cpu_result[i]);
            return false;
        }
    }
    return true;
}


// ============================================================
// MAIN
// ============================================================
int main() {
    printf("==========================================================\n");
    printf("  STEP 1: Naive Matrix Multiplication (Global Memory)\n");
    printf("  Matrix: %d x %d  *  %d x %d  =  %d x %d\n", M, K, K, N, M, N);
    printf("  Block size: %d x %d = %d threads\n", BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * BLOCK_SIZE);
    printf("==========================================================\n\n");

    // --- Allocate host (CPU) memory ---
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(size_A);     // host copy of A
    float *h_B = (float*)malloc(size_B);     // host copy of B
    float *h_C = (float*)malloc(size_C);     // will hold GPU result
    float *h_C_ref = (float*)malloc(size_C); // will hold CPU result

    // Fill A and B with random floats between 0 and 1
    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;

    // Compute CPU reference answer
    printf("Computing CPU reference result...\n");
    matmul_cpu(h_A, h_B, h_C_ref, M, K, N);
    printf("CPU done.\n\n");

    // --- Allocate device (GPU) memory ---
    // cudaMalloc allocates memory on the GPU's VRAM
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // --- Copy input data from CPU RAM to GPU VRAM ---
    // This goes over the PCIe bus — relatively slow, but only done once
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // --- Set up grid and block dimensions ---
    // Block: 16x16 threads
    // Grid: enough blocks to cover the full matrix
    //   gridDim.x = ceil(1024 / 16) = 64 blocks in x
    //   gridDim.y = ceil(1024 / 16) = 64 blocks in y
    //   Total: 64 * 64 = 4096 blocks * 256 threads = 1,048,576 threads
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("Grid: %d x %d blocks\n", gridDim.x, gridDim.y);
    printf("Total threads: %d\n\n", gridDim.x * gridDim.y * BLOCK_SIZE * BLOCK_SIZE);

    // --- Warm-up run (first launch has driver overhead) ---
    matmul_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    // --- Timed runs ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int NUM_RUNS = 10;
    cudaEventRecord(start);
    for (int i = 0; i < NUM_RUNS; i++) {
        matmul_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / NUM_RUNS;

    // --- Copy result back from GPU to CPU ---
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // --- Verify correctness ---
    bool correct = verify(h_C, h_C_ref, M * N);
    printf("Correctness: %s\n\n", correct ? "PASS" : "FAIL");

    // --- Performance metrics ---
    // FLOPs: each output element needs K multiplies + K adds = 2K ops
    // Total: 2 * M * N * K
    double flops = 2.0 * M * N * K;
    double gflops = (flops / (avg_ms / 1000.0)) / 1e9;

    printf("Average time: %.3f ms\n", avg_ms);
    printf("Performance:  %.1f GFLOPS\n", gflops);

    // For reference: RTX 4050 Laptop peak is ~5,046 GFLOPS (FP32)
    // The naive kernel will be FAR below this — that's the whole point!
    printf("\nYour RTX 4050 peak FP32: ~5046 GFLOPS\n");
    printf("Efficiency: %.2f%%\n", (gflops / 5046.0) * 100.0);

    // --- Cleanup ---
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
