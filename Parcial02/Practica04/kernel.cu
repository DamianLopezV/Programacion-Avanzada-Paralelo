#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <iostream>
#include <ctime>

const int sideSize = 32;

struct simple {
    int a;
    int b;
};

struct arreglo {
    int a[sideSize];
    int b[sideSize];
};

__global__ void aos(simple* a, simple* b)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < sideSize) {
        b[gid].a = a[gid].a + 6;
        b[gid].b = a[gid].b + 8;
    }
}

__global__ void soa(arreglo* a, arreglo* b)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < sideSize) {
        b->a[gid] = a->a[gid] + 10;
        b->b[gid] = a->b[gid] + 12;
    }
}

int main()
{
    simple a1[sideSize];
    simple a2[sideSize];

    for (int i = 0; i < sideSize; i++) {
        a1[i].a = 1;
        a1[i].b = 2;
    }

    simple* d_a1, *d_a2;
    cudaMalloc(&d_a1, sizeof(simple) * sideSize);
    cudaMalloc(&d_a2, sizeof(simple) * sideSize);

    cudaMemcpy(d_a1, a1, sizeof(simple) * sideSize, cudaMemcpyHostToDevice);

    arreglo* b1, * b2;
    b1 = (arreglo*)malloc(sizeof(arreglo));
    b2 = (arreglo*)malloc(sizeof(arreglo));

    for (int i = 0; i < sideSize; i++) {
        b1->a[i] = 3;
        b1->b[i] = 4;
    }

    arreglo* d_b1, * d_b2;
    cudaMalloc(&d_b1, sizeof(arreglo));
    cudaMalloc(&d_b2, sizeof(arreglo));

    cudaMemcpy(d_b1, b1, sizeof(arreglo), cudaMemcpyHostToDevice);

    dim3 block(1);
    dim3 grid(sideSize);

    aos << <grid, block >> > (d_a1, d_a2);
    soa << <grid, block >> > (d_b1, d_b2);

    cudaMemcpy(a2, d_a2, sizeof(simple) * sideSize, cudaMemcpyDeviceToHost);

    printf("AOS:\n");
    for (int i = 0; i < sideSize; i++) {
        printf("%d ", a2[i].a);
    }
    printf("\n");
    for (int i = 0; i < sideSize; i++) {
        printf("%d ", a2[i].b);
    }
    printf("\n");

    cudaMemcpy(b2, d_b2, sizeof(arreglo), cudaMemcpyDeviceToHost);

    printf("SOA:\n");
    for (int i = 0; i < sideSize; i++) {
        printf("%d ", b2->a[i]);
    }
    printf("\n");
    for (int i = 0; i < sideSize; i++) {
        printf("%d ", b2->b[i]);
    }
    printf("\n");

    return 0;
}