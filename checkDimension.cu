#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void checkIndex() {
    printf("[device] gridDim: (%d, %d, %d), "
        "[device] blockDim: (%d, %d, %d), "
        "[device] blockIdx: (%d, %d, %d), "
        "[device] threadIdx: (%d, %d, %d)\n"
        , gridDim.x, gridDim.y, gridDim.z
        , blockDim.x, blockDim.y, blockDim.z
        , blockIdx.x, blockIdx.y, blockIdx.z
        , threadIdx.x, threadIdx.y, threadIdx.z);
}

int main() {
    // define total data element
    int nElem = 12;

    // define grid and blok structure
    dim3 block(3, 1, 1);
    dim3 grid((nElem + block.x - 1) / block.x, 1, 1);

    // check grid and block dimension from host side
    printf("[host] gridDim: (%d, %d, %d)\n", grid.x, grid.y, grid.y);
    printf("[host] blockDim: (%d, %d, %d)\n", block.x, block.y, block.y);

    // check grid and block dimension from device side
    checkIndex << <grid, block >> > ();

    // reset device before you leave
    cudaDeviceReset();
}

/*
output:
[host] gridDim: (4, 1, 1)
[host] blockDim: (3, 1, 1)
[device] gridDim: (4, 1, 1), [device] blockDim: (3, 1, 1), [device] blockIdx: (3, 0, 0), [device] threadIdx: (0, 0, 0)
[device] gridDim: (4, 1, 1), [device] blockDim: (3, 1, 1), [device] blockIdx: (3, 0, 0), [device] threadIdx: (1, 0, 0)
[device] gridDim: (4, 1, 1), [device] blockDim: (3, 1, 1), [device] blockIdx: (3, 0, 0), [device] threadIdx: (2, 0, 0)
[device] gridDim: (4, 1, 1), [device] blockDim: (3, 1, 1), [device] blockIdx: (1, 0, 0), [device] threadIdx: (0, 0, 0)
[device] gridDim: (4, 1, 1), [device] blockDim: (3, 1, 1), [device] blockIdx: (1, 0, 0), [device] threadIdx: (1, 0, 0)
[device] gridDim: (4, 1, 1), [device] blockDim: (3, 1, 1), [device] blockIdx: (1, 0, 0), [device] threadIdx: (2, 0, 0)
[device] gridDim: (4, 1, 1), [device] blockDim: (3, 1, 1), [device] blockIdx: (0, 0, 0), [device] threadIdx: (0, 0, 0)
[device] gridDim: (4, 1, 1), [device] blockDim: (3, 1, 1), [device] blockIdx: (0, 0, 0), [device] threadIdx: (1, 0, 0)
[device] gridDim: (4, 1, 1), [device] blockDim: (3, 1, 1), [device] blockIdx: (0, 0, 0), [device] threadIdx: (2, 0, 0)
[device] gridDim: (4, 1, 1), [device] blockDim: (3, 1, 1), [device] blockIdx: (2, 0, 0), [device] threadIdx: (0, 0, 0)
[device] gridDim: (4, 1, 1), [device] blockDim: (3, 1, 1), [device] blockIdx: (2, 0, 0), [device] threadIdx: (1, 0, 0)
[device] gridDim: (4, 1, 1), [device] blockDim: (3, 1, 1), [device] blockIdx: (2, 0, 0), [device] threadIdx: (2, 0, 0)
*/
