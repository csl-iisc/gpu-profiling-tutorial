// gemm_tiled.cu ── shared-memory tiled GEMM, one NVTX range
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <iostream>
#include <chrono>

#ifndef BLOCK
#define BLOCK 32
#endif

__global__ void matmul_tiled(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* C, int N)
{
    __shared__ float As[BLOCK][BLOCK];
    __shared__ float Bs[BLOCK][BLOCK];

    int row = blockIdx.y * BLOCK + threadIdx.y;
    int col = blockIdx.x * BLOCK + threadIdx.x;

    float acc = 0.f;
    for (int t = 0; t < N; t += BLOCK) {
        As[threadIdx.y][threadIdx.x] = A[row * N + t + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        __syncthreads();

        for (int k = 0; k < BLOCK; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = acc;
}

int main(int argc, char** argv)
{
    const int N = (argc > 1) ? std::stoi(argv[1]) : 1024;
    const size_t bytes = size_t(N) * N * sizeof(float);

    float *hA, *hB, *hC;
    cudaHostAlloc(&hA, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&hB, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&hC, bytes, cudaHostAllocDefault);
    for (int i = 0; i < N * N; ++i) hA[i] = hB[i] = 1.f;

    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes); cudaMalloc(&dB, bytes); cudaMalloc(&dC, bytes);

    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK);

    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    nvtxRangePushA("gemm tiled kernel");
    matmul_tiled<<<grid, block>>>(dA, dB, dC, N);
    nvtxRangePop();
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
    std::cout << "C[0] = " << hC[0] << '\n';

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaFreeHost(hA); cudaFreeHost(hB); cudaFreeHost(hC);
    return 0;
}
