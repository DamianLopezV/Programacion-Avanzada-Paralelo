
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

const int size = 1024;
const int stream_Size = 8;
__global__ void addKernel(int *a, int *b, int *c)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < size) {
        c[gid] = a[gid] + b[gid];
    }
}

int main()
{
    int a[size], b[size], c[size];
    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    int* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, sizeof(int) * size);
    cudaMalloc((void**)&d_b, sizeof(int) * size);
    cudaMalloc((void**)&d_c, sizeof(int) * size);

    dim3 block(128);
    dim3 grid(size / block.x);
    cudaStream_t str[stream_Size];
    for (int i = 0; i < stream_Size; i++) {
        cudaStreamCreate(&str[i]);
    }

    for (int i = 0; i < stream_Size; i++) {
        cudaMemcpyAsync(d_a, a, sizeof(int) * size, cudaMemcpyHostToDevice, str[i]);
        cudaMemcpyAsync(d_b, b, sizeof(int) * size, cudaMemcpyHostToDevice, str[i]);
        addKernel << <grid, block, 0, str[i] >> > (d_a, d_b, d_c);
        cudaMemcpyAsync(c, d_c, sizeof(int) * size, cudaMemcpyDeviceToHost, str[i]);
    }

    for (int i = 0; i < size; i++) {
        printf("%d: %d\n", i, c[i]);
    }
    

    return 0;
}


