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

__device__ float devData;
__global__ void checkGlobalVariable() {
    // display the original value
    printf("[Device] devData : %f\n", devData);

    // alter the value
    devData += 2.0f;
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
    float value = 3.14f;
    cudaMemcpyToSymbol(devData, &value, sizeof(float));
    printf("[host] value = %f\n", value);

    // invoke the kernel
    checkGlobalVariable << <1, 1 >> > ();

    // copy the global variable back to the host
    cudaMemcpyFromSymbol(&value, devData, sizeof(float));
    printf("[host] value = %f\n", value);

    // reset device
    CHECK(cudaDeviceReset());
}

/*
output:
[host] C:\coding\Cuda\x64\Debug\Cuda.exe Starting...
[host] Using Device 0: NVIDIA GeForce MX450
[host] value = 3.140000
[Device] devData : 3.140000
[host] value = 5.140000
*/
