#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<string.h>

// error check
#define CHECK(call) \
{ \
    const cudaError_t error = call;\
    if (error != cudaSuccess) { \
        printf("[device] Error: %s %d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(-10*error); \
    } \
}

__global__ void nestedHelloWorld(int const iSize, int iDepth) {
    int tid = threadIdx.x;
    printf("[Device] Hello World = (recursion depth: %d), (block: %d), (thread: %d)\n",
        iDepth, blockIdx.x, threadIdx.x);

    // condition to stop recursive execution
    if (iSize == 1) return;

    // reduce block size to half
    int nthreads = iSize >> 1;

    // thread 0 launches child grid recursively
    if (tid == 0 && nthreads > 0) {
        nestedHelloWorld << <1, nthreads >> > (nthreads, ++iDepth);
        printf("[Device] --------> nested execution depth : %d\n", iDepth);
    }
}

int main(int argc, char** argv) {
    printf("[host] %s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("[host] Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // data size
    int size = 8;
    dim3 block(size, 1);
    dim3 grid(1, 1);
    printf("[host] Execution configure : grid(%d, %d), block(%d, %d)\n",
        grid.x, grid.y, block.x, block.y);

    // kernel: nestedHelloWorld
    nestedHelloWorld << <grid, block >> > (size, 0);

    // reset device
    CHECK(cudaDeviceReset());
}

/*
output:
*/
