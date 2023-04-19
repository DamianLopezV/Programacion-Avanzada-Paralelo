#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <iostream>
#include <ctime>

const int sideSize = 32;
const int arraySize = sideSize * sideSize;

__global__ void search(int* a, int* b, int* c)
{
    __shared__ int d[1];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int bid = blockIdx.x + blockIdx.y * gridDim.x;
    int blockSize = blockDim.x * blockDim.y;
    int gid = tid + bid * blockSize;

    if (a[gid] == *b) {
        d[0] = gid;
    }
    __syncthreads();
    c[0] = d[0];
}

int main()
{
    int a[arraySize];
    int b = 88;
    int c[1];

    for (int i = 0; i < arraySize; i++) {
        a[i] = i * 2;
    }

    int* d_a, * d_b, * d_c;

    cudaMalloc((void**)&d_a, arraySize * sizeof(int));
    cudaMalloc((void**)&d_b, sizeof(int));
    cudaMalloc((void**)&d_c, sizeof(int));

    cudaMemcpy(d_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(1);
    //dim3 block(arraySize);
    dim3 block(sideSize, sideSize);

    clock_t gpu_start, gpu_stop;
    gpu_start = clock();

    search << <grid, block >> > (d_a, d_b, d_c);

    gpu_stop = clock();

    cudaMemcpy(c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

    printf("a:\n\r");
    for (int i = 0; i < arraySize; i++)
    {
        printf("%d ", a[i]);
    }
    printf("\n\r");

    printf("c:\n\r");
    for (int i = 0; i < 1; i++)
    {
        printf("%d ", c[i]);
    }
    printf("\n\r");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
