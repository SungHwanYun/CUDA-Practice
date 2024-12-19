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

    // allocate the pinned host memory
    float* h_A;
    CHECK(cudaMallocHost((void**)&h_A, nbytes));

    // allocate the device memory
    float* d_A;
    CHECK(cudaMalloc((void**)&d_A, nbytes));

    // initialize the host memory
    for (unsigned int i = 0; i < size; i++) h_A[i] = 0.5f;

    // transfer data from host to the device
    CHECK(cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice));

    // transfer data from the device to the host
    CHECK(cudaMemcpy(h_A, d_A, nbytes, cudaMemcpyDeviceToHost));

    // free memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFreeHost(h_A));

    // reset device
    CHECK(cudaDeviceReset());
}

/*
output:
c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe
[host] ./Cuda.exe Starting...
==22052== NVPROF is profiling process 22052, command: ./Cuda.exe
[host] Using Device 0: NVIDIA GeForce MX450
==22052== Profiling application: ./Cuda.exe
==22052== Warning: 23 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==22052== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   51.82%  5.5341ms         1  5.5341ms  5.5341ms  5.5341ms  [CUDA memcpy HtoD]
                   48.18%  5.1446ms         1  5.1446ms  5.1446ms  5.1446ms  [CUDA memcpy DtoH]
      API calls:   65.09%  70.869ms         1  70.869ms  70.869ms  70.869ms  cudaSetDevice
                   20.68%  22.519ms         1  22.519ms  22.519ms  22.519ms  cudaDeviceReset
                    9.97%  10.857ms         2  5.4287ms  5.2155ms  5.6419ms  cudaMemcpy
                    1.98%  2.1516ms         1  2.1516ms  2.1516ms  2.1516ms  cudaMallocHost
                    1.69%  1.8347ms         1  1.8347ms  1.8347ms  1.8347ms  cudaFreeHost
                    0.32%  345.60us         1  345.60us  345.60us  345.60us  cudaFree
                    0.22%  239.10us         1  239.10us  239.10us  239.10us  cudaMalloc
                    0.04%  46.000us       114     403ns       0ns  23.500us  cuDeviceGetAttribute
                    0.00%  3.8000us         1  3.8000us  3.8000us  3.8000us  cuModuleGetLoadingMode
                    0.00%  3.6000us         1  3.6000us  3.6000us  3.6000us  cudaGetDeviceProperties
                    0.00%  2.7000us         3     900ns     100ns  2.4000us  cuDeviceGetCount
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuDeviceTotalMem
                    0.00%  1.0000us         2     500ns       0ns  1.0000us  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
*/
