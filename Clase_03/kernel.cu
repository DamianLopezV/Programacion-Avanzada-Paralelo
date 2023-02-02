#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
/*
__global__ void idx_calc_tid(int* input)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int blockSize = blockDim.x * blockDim.y * blockDim.z;
    int blockOffset = blockSize * blockIdx.x;
    int rowOffset = blockSize * blockIdx.y * gridDim.x;
    int depthOffset = blockSize * blockIdx.z * gridDim.x * gridDim.y;
    int gid = tid + blockOffset + rowOffset + depthOffset;
    printf("[Device] blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d, tid: %d, gid: %d, data: %d\n\r", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, tid, gid, input[gid]);
}*/

/*
__global__ void idx_calc_tid(int* d_a, int* d_b, int* d_c)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int blockSize = blockDim.x * blockDim.y * blockDim.z;
    int gid = tid + bid * blockSize;

    if (gid < 10000) {
        d_c[gid] = d_a[gid] + d_b[gid];
    }
}*/

__global__ void idx_calc_tid(int* d_a, int* d_b, int* d_c, int* d_d)
{
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int blockSize = blockDim.x * blockDim.y * blockDim.z;
    int gid = tid + bid * blockSize;

    if (gid < 10000) {
        d_d[gid] = d_a[gid] + d_b[gid] + d_c[gid];
    }
}

int main()
{
    
    const int vectorSize = 10000;
    dim3 grid(8, 8, 8);
    dim3 block(8, 4, 4);
    int a[vectorSize], b[vectorSize], c[vectorSize], d[vectorSize];
    for (int i = 0; i < vectorSize; i++) {
        a[i] = rand() % 256;
        b[i] = rand() % 256;
        c[i] = rand() % 256;
    }
    int* d_a, * d_b, * d_c, * d_d;
    cudaMalloc((void**)&d_a, vectorSize * sizeof(int));
    cudaMalloc((void**)&d_b, vectorSize * sizeof(int));
    cudaMalloc((void**)&d_c, vectorSize * sizeof(int));
    cudaMalloc((void**)&d_d, vectorSize * sizeof(int));
    cudaMemcpy(d_a, a, vectorSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, vectorSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, vectorSize * sizeof(int), cudaMemcpyHostToDevice);

    clock_t gpu_start, gpu_stop;
    gpu_start = clock();

    idx_calc_tid << <grid, block >> > (d_a, d_b, d_c, d_d);
    cudaMemcpy(d, d_d, vectorSize * sizeof(int), cudaMemcpyDeviceToHost);

    gpu_stop = clock();
    /*
    for (int i = 0; i < vectorSize; i++) {
        printf("a: %d + b: %d + c: %d = d: %d\n\r", a[i], b[i], c[i], d[i]);
    }*/

    double cps_gpu = (double)((double)gpu_stop - gpu_start / CLOCKS_PER_SEC);
    printf("time: %4.6f\n\r", cps_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);

    return 0;
}
