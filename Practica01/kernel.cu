
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void printKernel()
{
    printf("Thread ID X: %d Thread ID Y: %d Thread ID Z: %d", threadIdx.x, threadIdx.y, threadIdx.z);
}

int main()
{
    dim3 block(2,2,2);
    dim3 grid(4,4,4);
    printKernel<< <grid, block >> > ();

    return 0;
}

