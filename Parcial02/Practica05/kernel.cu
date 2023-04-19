
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <iostream>
#include <ctime>

const int sideSize = 64;
const int arraySize = sideSize * sideSize;
const int threadSize = 32;
const int threadMaxSize = threadSize * threadSize;

__global__ void unrollingTranspose(int* a, int* b)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int bid = blockIdx.x + blockIdx.y * gridDim.x;
    int blockSize = blockDim.x * blockDim.y;
    int gid = tid + bid * blockSize;

    int gid2 = gid % sideSize * sideSize + gid / sideSize;
    int offset = threadSize / 2;

    for (int i = 0; i < (arraySize + threadMaxSize - 1) / threadMaxSize; i+=2)
    {
        if (gid + threadMaxSize * i < arraySize) {
            b[gid2 + offset * i] = a[gid + threadMaxSize * i];
        }
        if (gid + threadMaxSize * i + threadMaxSize < arraySize) {
            b[gid2 + offset * i + offset] = a[gid + threadMaxSize * i + threadMaxSize];
        }
    }

}

int main()
{


    int a[arraySize];
    int b[arraySize];

    for (int i = 0; i < arraySize; i++) {
        a[i] = rand() % 255;
    }

    int* d_a, * d_b;

    cudaMalloc((void**)&d_a, arraySize * sizeof(int));
    cudaMalloc((void**)&d_b, arraySize * sizeof(int));
    cudaMemcpy(d_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(1);
    dim3 block(threadSize, threadSize);

    clock_t gpu_start, gpu_stop;
    gpu_start = clock();

    unrollingTranspose << <grid, block >> > (d_a, d_b);

    gpu_stop = clock();

    cudaMemcpy(b, d_b, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    printf("a:\n\r");
    for (int i = 0; i < sideSize; i++)
    {
        for (int j = 0; j < sideSize; j++)
        {
            printf("%d ", a[i * sideSize + j]);
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

    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}