
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
/*
__global__ void idx_calc_tid(int *input)
{
    int tid = threadIdx.x;
    int offset = blockDim.x * blockIdx.x;
    int gid = tid + offset;
    printf("[Device] blockIdx:x: %d, threadIdx.x: %d, gid: %d, data: %d\n\r", blockIdx.x, tid, gid, input[tid]);
}*/

/*__global__ void idx_calc_tid(int* input)
{
    int tid = threadIdx.x;
    int blockOffset = blockDim.x * blockIdx.x;
    int rowOffset = blockDim.x * blockIdx.y * gridDim.x;
    int gid = tid + blockOffset + rowOffset;
    printf("[Device] blockIdx.x: %d, blockIdx.y: %d, threadIdx.x: %d, gid: %d, data: %d\n\r", blockIdx.x, blockIdx.y, tid, gid, input[tid]);
}*/

__global__ void idx_calc_tid(int* input)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int blockSize = blockDim.x * blockDim.y;
    int blockOffset = blockSize * blockIdx.x;
    int rowOffset = blockSize * blockIdx.y * gridDim.x;
    int gid = tid + blockOffset + rowOffset;
    printf("[Device] blockIdx.x: %d, blockIdx.y: %d, threadIdx.x: %d, threadIdx.y: %d, tid: %d, gid: %d, data: %d\n\r", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, tid, gid, input[gid]);
}

int main()
{
    const int vectorSize = 32;
    dim3 grid(2, 2);
    dim3 block(4, 2);
    int a[vectorSize];
    for (int i = 0; i < vectorSize; i++) {
        a[i] = i;
    }
    int* d_a;
    cudaMalloc((void**)&d_a, vectorSize * sizeof(int));
    cudaMemcpy(d_a, a, vectorSize * sizeof(int), cudaMemcpyHostToDevice);
    idx_calc_tid << <grid, block >> > (d_a);

    cudaFree(d_a);

    return 0;
}

