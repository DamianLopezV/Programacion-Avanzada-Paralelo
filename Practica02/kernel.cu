
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void multiplicationKernel(int* d_a, int* d_b, int* d_c)
{
    d_c[threadIdx.x] = d_a[threadIdx.x] * d_b[threadIdx.x];
}

int main()
{
    const int N = 3;
    int a[N] = { 1,0,1 };
    int b[N] = { 2,4,3 };
    int c[N] = { 0 };
    int size = N * sizeof(int);
    int* d_a, * d_b, * d_c;

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    multiplicationKernel << <1, 3 >> > (d_a, d_b, d_c);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    printf("[1,0,1]*[2,4,3] = [%d,%d,%d]", c[0], c[1], c[2]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

