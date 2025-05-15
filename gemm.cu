// naive_gemm.cu  ── plain row-major C = A × B (O(N3))
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void matmul_naive(const float* A, const float* B,
                             float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.f;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

int main(int argc, char** argv)
{
    const int N = (argc > 1) ? std::stoi(argv[1]) : 1024;
    const size_t bytes = size_t(N) * N * sizeof(float);

    float *hA = new float[N * N], *hB = new float[N * N], *hC = new float[N * N]; 

    for (int i = 0; i < N * N; ++i) hA[i] = hB[i] = 1.f;

    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    matmul_naive<<<grid, block>>>(dA, dB, dC, N);

    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";

    std::cout << "C[0] = " << hC[0] << '\n';         // quick check (== N)
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] hC;
    return 0;
}
