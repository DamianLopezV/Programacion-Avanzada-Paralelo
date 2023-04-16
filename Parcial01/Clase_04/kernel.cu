
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <iostream>
#include <ctime>

using namespace std;

#define GPUErrorAssertion(ans) {gpuAssert((ans),__FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n\r", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void addMatrix(int* a, int* b, int* c) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int blockSize = blockDim.x * blockDim.y * blockDim.z;
    int gid = tid + bid * blockSize;

    for (int i = 0; i < 2; i++)
    {
        c[gid] += a[(gid / 2) * 2 + i] * b[gid % 2 + i * 2];
        printf("c: %d, a: %d, b: %d\n", gid, (gid / 2) * 2 + i, gid % 2 + i * 2);
    }
}

int main()
{
    const int matrixSize = 4, sideSize = 2;
    int a[matrixSize], int b[matrixSize], int c[matrixSize];

    for (int i = 0; i < sideSize; i++)
    {
        for (int j = 0; j < sideSize; j++)
        {
            a[i * sideSize + j] = rand() % 10;
            b[i * sideSize + j] = rand() % 10;
            c[i * sideSize + j] = 0;
        }
    }

    int* d_a, * d_b,* d_c;
    (cudaMalloc((void**)&d_a, matrixSize * sizeof(int)));
    (cudaMalloc((void**)&d_b, matrixSize * sizeof(int)));
    (cudaMalloc((void**)&d_c, matrixSize * sizeof(int)));
    (cudaMemcpy(d_a, a, matrixSize * sizeof(int), cudaMemcpyHostToDevice));
    (cudaMemcpy(d_b, b, matrixSize * sizeof(int), cudaMemcpyHostToDevice));

    dim3 grid(1);
    dim3 block(4);

    clock_t gpu_start, gpu_stop;
    gpu_start = clock();

    addMatrix << <grid, block >> > (d_a, d_b, d_c);

    gpu_stop = clock();

    cudaMemcpy(c, d_c, matrixSize * sizeof(int), cudaMemcpyDeviceToHost);
    printf("a:\n\r");
    for (int i = 0; i < sideSize; i++)
    {
        for (int j = 0; j < sideSize; j++)
        {
            printf("%d ",a[i * sideSize + j]);
        }
        printf("\n\r");
    }
    printf("b:\n\r");
    for (int i = 0; i < sideSize; i++)
    {
        for (int j = 0; j < sideSize; j++)
        {
            printf("%d ", b[i * sideSize + j]);
        }
        printf("\n\r");
    }
    printf("c:\n\r");
    for (int i = 0; i < sideSize; i++)
    {
        for (int j = 0; j < sideSize; j++)
        {
            printf("%d ", c[i * sideSize + j]);
        }
        printf("\n\r");
    }
}


