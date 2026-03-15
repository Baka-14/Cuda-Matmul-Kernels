#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define M 1024
#define K 1024
#define N 1024
#define TILE_SIZE 16


// ============================================================
// TILED KERNEL: Threads cooperate to load data into shared memory
//
// Shared memory is a small, fast scratchpad (~100x faster than
// global memory) that all threads in a block can access.
//
// Instead of each thread reading 1024 floats from global memory,
// the block loads a 16x16 tile at a time. Each thread loads just
// ONE element, then all 256 threads read from shared memory.
// This reuses each global memory read 16 times!
// ============================================================
__global__ void matmul_tiled(float *A, float *B, float *C,
                             int m, int k, int n) {

    // Declare shared memory — these live on-chip, shared by the block
    // Think of them as two 16x16 whiteboards for the block
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Which element of C does this thread compute?
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Accumulator for the dot product (builds up across all tiles)
    float sum = 0.0f;

    // How many tiles do we need to cover the K dimension?
    // 1024 / 16 = 64 tiles
    int numTiles = (k + TILE_SIZE - 1) / TILE_SIZE;

    // Slide through tiles one at a time
    for (int t = 0; t < numTiles; t++) {

        // === PHASE 1: Load one tile of A and B into shared memory ===
        // Each thread loads exactly ONE element — 256 threads load
        // the full 16x16 tile cooperatively.

        // Load A[row][t*TILE_SIZE + threadIdx.x] into tileA
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (row < m && aCol < k) {
            tileA[threadIdx.y][threadIdx.x] = A[row * k + aCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B[t*TILE_SIZE + threadIdx.y][col] into tileB
        int bRow = t * TILE_SIZE + threadIdx.y;
        if (bRow < k && col < n) {
            tileB[threadIdx.y][threadIdx.x] = B[bRow * n + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // === BARRIER 1: Wait for ALL threads to finish loading ===
        // Without this, thread 5 might try to read tileA[3][7]
        // before thread 3 has written it. RACE CONDITION!
        __syncthreads();

        // === PHASE 2: Compute partial dot product from shared memory ===
        // This is the FAST part — shared memory reads are ~100x faster
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        // === BARRIER 2: Wait before loading the NEXT tile ===
        // Without this, fast threads might overwrite tileA/tileB
        // with the next tile while slow threads are still reading!
        __syncthreads();
    }

    // Write final result
    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
}


// ============================================================
// CPU reference and verification (same as before)
// ============================================================
void matmul_cpu(float *A, float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++)
                sum += A[i * k + p] * B[p * n + j];
            C[i * n + j] = sum;
        }
}

bool verify(float *gpu, float *cpu, int size) {
    for (int i = 0; i < size; i++)
        if (fabs(gpu[i] - cpu[i]) > 1e-2) {
            printf("MISMATCH at %d: GPU=%.6f CPU=%.6f\n", i, gpu[i], cpu[i]);
            return false;
        }
    return true;
}


int main() {
    printf("==========================================================\n");
    printf("  STEP 2: Tiled Matrix Multiplication (Shared Memory)\n");
    printf("  Matrix: %d x %d  *  %d x %d  =  %d x %d\n", M, K, K, N, M, N);
    printf("  Tile size: %d x %d\n", TILE_SIZE, TILE_SIZE);
    printf("==========================================================\n\n");

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    float *h_C_ref = (float*)malloc(size_C);

    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;

    printf("Computing CPU reference...\n");
    matmul_cpu(h_A, h_B, h_C_ref, M, K, N);
    printf("CPU done.\n\n");

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    // Warm-up
    matmul_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    // Timed runs
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int NUM_RUNS = 10;
    cudaEventRecord(start);
    for (int i = 0; i < NUM_RUNS; i++) {
        matmul_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / NUM_RUNS;

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    bool correct = verify(h_C, h_C_ref, M * N);

    double flops = 2.0 * M * N * K;
    double gflops = (flops / (avg_ms / 1000.0)) / 1e9;

    printf("Correctness:  %s\n", correct ? "PASS" : "FAIL");
    printf("Average time: %.3f ms\n", avg_ms);
    printf("Performance:  %.1f GFLOPS\n", gflops);
    printf("\n--- Comparison ---\n");
    printf("Naive kernel:  661.0 GFLOPS (13.1%% efficiency)\n");
    printf("Tiled kernel:  %.1f GFLOPS (%.2f%% efficiency)\n",
           gflops, (gflops / 5046.0) * 100.0);
    printf("Speedup:       %.2fx\n", gflops / 661.0);

    // Shared memory usage info
    printf("\nShared memory per block: %lu bytes (2 tiles x %dx%d x 4 bytes)\n",
           2 * TILE_SIZE * TILE_SIZE * sizeof(float), TILE_SIZE, TILE_SIZE);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
