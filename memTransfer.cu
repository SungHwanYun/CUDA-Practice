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
    float *h_A = (float *)malloc(nbytes);

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
==11068== NVPROF is profiling process 11068, command: ./Cuda.exe
[host] Using Device 0: NVIDIA GeForce MX450
[host] value = 3.140000
[Device] devData : 3.140000
[host] value = 5.140000
==11068== Profiling application: ./Cuda.exe
==11068== Warning: 29 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==11068== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   93.63%  36.192us         1  36.192us  36.192us  36.192us  checkGlobalVariable(void)
                    4.64%  1.7920us         1  1.7920us  1.7920us  1.7920us  [CUDA memcpy DtoH]
                    1.74%     672ns         1     672ns     672ns     672ns  [CUDA memcpy HtoD]
      API calls:   69.89%  82.003ms         1  82.003ms  82.003ms  82.003ms  cudaSetDevice
                   26.27%  30.823ms         1  30.823ms  30.823ms  30.823ms  cudaDeviceReset
                    3.22%  3.7805ms         1  3.7805ms  3.7805ms  3.7805ms  cudaMemcpyToSymbol
                    0.42%  488.90us         1  488.90us  488.90us  488.90us  cudaMemcpyFromSymbol
                    0.16%  189.60us         1  189.60us  189.60us  189.60us  cudaLaunchKernel
                    0.02%  20.200us       114     177ns       0ns  3.0000us  cuDeviceGetAttribute
                    0.01%  16.800us         1  16.800us  16.800us  16.800us  cuLibraryUnload
                    0.00%  3.7000us         1  3.7000us  3.7000us  3.7000us  cudaGetDeviceProperties
                    0.00%  2.3000us         3     766ns     100ns  2.0000us  cuDeviceGetCount
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuDeviceTotalMem
                    0.00%  1.6000us         1  1.6000us  1.6000us  1.6000us  cuModuleGetLoadingMode
                    0.00%  1.0000us         2     500ns     100ns     900ns  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid
*/
