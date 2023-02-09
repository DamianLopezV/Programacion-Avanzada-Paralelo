
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <iostream>
#include <ctime>

const int sideSize = 32;
const int arraySize = sideSize * sideSize;

__global__ void addKernel(int *a, int *b, int *c)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int blockSize = blockDim.x * blockDim.y * blockDim.z;
    int gid = tid + bid * blockSize;

    for (int i = -1; i < 2; i++)
    {
        for (int j = -1; j < 2; j++)
        {
            if (gid + i + j * sideSize >= 0 && gid + i + j * sideSize < arraySize && (gid + 1 + i) % sideSize != 0) {
                c[gid] += a[gid + i + j * sideSize] * b[(i + 1) + (j + 1) * 3];
            }
        }
    }
}

int main()
{
    

    int a[arraySize];
    int b[9] = { -1,-2,-1,0,0,0,1,2,1 };
    //int b[9] = { 0,0,0,0,1,0,0,0,0 };
    //int b[9] = { 0,0,0,1,1,1,0,0,0 };
    int c[arraySize];

    for (int i = 0; i < arraySize; i++) {
        a[i] = rand() % 255;
    }

    int* d_a, * d_b, * d_c;

    cudaMalloc((void**)&d_a, arraySize * sizeof(int));
    cudaMalloc((void**)&d_b, 9 * sizeof(int));
    cudaMalloc((void**)&d_c, arraySize * sizeof(int));
    cudaMemcpy(d_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, 9 * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(1);
    //dim3 block(arraySize);
    dim3 block(sideSize, sideSize);

    clock_t gpu_start, gpu_stop;
    gpu_start = clock();

    addKernel << <grid, block >> > (d_a, d_b, d_c);

    gpu_stop = clock();

    cudaMemcpy(c, d_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    printf("a:\n\r");
    for (int i = 0; i < sideSize; i++)
    {
        for (int j = 0; j < sideSize; j++)
        {
            printf("%d ", a[i * sideSize + j]);
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

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

