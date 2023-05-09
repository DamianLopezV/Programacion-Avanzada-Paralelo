
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

const int N = 1024;
__global__ void prefixSums(int *a, int*b)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = 0; i < N; i++) {
        if (gid + i < N) {
            b[gid + i] += a[gid];
        }
        __syncthreads();
    }
}

__global__ void prefixSums2(int* a, int* b)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    b[gid] = a[gid];
    for (int i = 1; i < N; i*=2) {
        if (gid + i < N) {
            if ((gid + 1) % i == 0) {
                b[gid + i] += b[gid + i / 2];
            }
        }
        __syncthreads();
    }
}

int main()
{
    int a[N];
    int b[N];
    int* d_a, * d_b;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));

    for (int i = 0; i < N; i++) {
        a[i] = 1;
        b[i] = 0;
        printf("%d ", a[i]);
    }
    printf("\n\n");

    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    
    prefixSums << <1, N >> > (d_a, d_b);

    cudaMemcpy(b, d_b, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("%d ", b[i]);
    }
    printf("\n\n");

    return 0;
}