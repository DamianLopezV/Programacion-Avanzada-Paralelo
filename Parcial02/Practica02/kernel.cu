#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <stdio.h>
#include <iostream>
#include <ctime>

const int sideSize = 32;
const int arraySize = sideSize * sideSize;

__global__ void bubbleSort(int* a)
{
    __shared__ int b[arraySize];

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int bid = blockIdx.x + blockIdx.y * gridDim.x;
    int blockSize = blockDim.x * blockDim.y;
    int gid = tid + bid * blockSize;

    b[gid] = a[gid];
    __syncthreads();
    int temp;
    for (int i = 0; i < arraySize / 2 + 1; i++) {
        if (gid * 2 < arraySize - 1 && b[gid * 2] > b[gid * 2 + 1]) {
                temp = b[gid * 2 + 1];
                b[gid * 2 + 1] = b[gid * 2];
                b[gid * 2] = temp;
        }
        __syncthreads();

        if (gid * 2 < arraySize - 2 && b[gid * 2 + 1] > b[gid * 2 + 2]) {
            temp = b[gid * 2 + 2];
            b[gid * 2 + 2] = b[gid * 2 + 1];
            b[gid * 2 + 1] = temp;
        }
        __syncthreads();
    }
    __syncthreads();
    a[gid] = b[gid];
}

int main()
{
    int a[arraySize];

    for (int i = 0; i < arraySize; i++) {
        a[i] = arraySize - i;
    }

    int* d_a;

    cudaMalloc((void**)&d_a, arraySize * sizeof(int));
    cudaMemcpy(d_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    printf("Unsorted:\n\r");
    for (int i = 0; i < arraySize; i++)
    {
        printf("%d ", a[i]);
    }
    printf("\n\r");

    dim3 grid(1);
    //dim3 block(arraySize);
    dim3 block(sideSize, sideSize);

    clock_t gpu_start, gpu_stop;
    gpu_start = clock();

    bubbleSort << <grid, block >> > (d_a);

    gpu_stop = clock();

    cudaMemcpy(a, d_a, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sorted:\n\r");
    for (int i = 0; i < arraySize; i++)
    {
        printf("%d ", a[i]);
    }
    printf("\n\r");


    return 0;
}