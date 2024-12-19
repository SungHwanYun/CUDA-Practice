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

int main(int argc, char** argv) {
    printf("[host] %s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("[host] Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // memory size
    unsigned int size = 1 << 22;
    unsigned int nbytes = size * sizeof(float);

    // allocate the host memory
    float* h_A = (float*)malloc(nbytes);

    // allocate the device memory
    float* d_A;
    cudaMalloc((void**)&d_A, nbytes);

    // initialize the host memory
    for (unsigned int i = 0; i < size; i++) h_A[i] = 0.5f;

    // transfer data from host to the device
    cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice);

    // transfer data from the device to the host
    cudaMemcpy(h_A, d_A, nbytes, cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_A);
    free(h_A);

    // reset device
    CHECK(cudaDeviceReset());
}

/*
output:
c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe
[host] ./Cuda.exe Starting...
==20568== NVPROF is profiling process 20568, command: ./Cuda.exe
[host] Using Device 0: NVIDIA GeForce MX450
==20568== Profiling application: ./Cuda.exe
==20568== Warning: 34 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==20568== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   52.14%  5.8245ms         1  5.8245ms  5.8245ms  5.8245ms  [CUDA memcpy HtoD]
                   47.86%  5.3469ms         1  5.3469ms  5.3469ms  5.3469ms  [CUDA memcpy DtoH]
      API calls:   65.23%  64.969ms         1  64.969ms  64.969ms  64.969ms  cudaSetDevice
                   22.25%  22.166ms         1  22.166ms  22.166ms  22.166ms  cudaDeviceReset
                   12.04%  11.997ms         2  5.9983ms  5.6629ms  6.3338ms  cudaMemcpy
                    0.25%  247.30us         1  247.30us  247.30us  247.30us  cudaFree
                    0.20%  194.90us         1  194.90us  194.90us  194.90us  cudaMalloc
                    0.02%  19.800us       114     173ns       0ns  3.5000us  cuDeviceGetAttribute
                    0.00%  3.8000us         1  3.8000us  3.8000us  3.8000us  cudaGetDeviceProperties
                    0.00%  2.5000us         3     833ns     100ns  2.2000us  cuDeviceGetCount
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuDeviceTotalMem
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuModuleGetLoadingMode
                    0.00%  1.0000us         2     500ns     100ns     900ns  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
*/
