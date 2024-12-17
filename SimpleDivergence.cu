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
__global__ void warmingup(float* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    if ((tid / warpSize) % 2 == 0) {
        a = 100.0f;
    }
    else {
        b = 200.0f;
    }
    c[tid] = a + b;
}
__global__ void mathKernel1(float* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    if (tid % 2 == 0) {
        a = 100.0f;
    }
    else {
        b = 200.f;
    }
    c[tid] = a + b;
}
__global__ void mathKernel2(float* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    if ((tid / warpSize) % 2 == 0) {
        a = 100.0f;
    }
    else {
        b = 200.0f;
    }
    c[tid] = a + b;
}
__global__ void mathKernel3(float* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    bool ipred = (tid % 2 == 0);
    if (ipred) {
        ia = 100.0f;
    }
    if (!ipred) {
        ib = 200.0f;
    }
    c[tid] = ia + ib;
}
__global__ void mathKernel4(float* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    int itid = tid >> 5;
    if (itid & 0x01 == 0) {
        ia = 100.0f;
    } else {
        ib = 200.0f;
    }
    c[tid] = ia + ib;
}
int main(int argc, char **argv) {
    printf("[host] %s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("[host] Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size
    int size = 64;
    int blocksize = 64;
    if (argc > 1) blocksize = atoi(argv[1]);
    if (argc > 2) size = atoi(argv[2]);
    printf("[host] Data size : %d\n", size);

    // set up execution configuration
    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("[host] Execution Configure (block %d, grid %d)\n", block.x, grid.x);

    // allocate gpu memory
    float* d_C;
    size_t nBytes = size * sizeof(float);
    cudaMalloc((float**) &d_C, nBytes);

    // run a warmup kernel to remove overhead
    warmingup << <grid, block >> > (d_C);
    cudaDeviceSynchronize();
    printf("[host] warmup <<<%4d, %4d>>>\n", grid.x, block.x);

    // run kernel 1
    mathKernel1 << <grid, block >> > (d_C);
    cudaDeviceSynchronize();
    printf("[host] mathKernel1 <<<%4d, %4d>>>\n", grid.x, block.x);

    // run kernel 2
    mathKernel2 << <grid, block >> > (d_C);
    cudaDeviceSynchronize();
    printf("[host] mathKernel2 <<<%4d, %4d>>>\n", grid.x, block.x);

    // run kernel 3
    mathKernel3 << <grid, block >> > (d_C);
    cudaDeviceSynchronize();
    printf("[host] mathKernel3 <<<%4d, %4d>>>\n", grid.x, block.x);

    // run kernel 4
    mathKernel4 << <grid, block >> > (d_C);
    cudaDeviceSynchronize();
    printf("[host] mathKernel4 <<<%4d, %4d>>>\n", grid.x, block.x);

    // free gpu memory and reset device
    cudaFree(d_C);
    cudaDeviceReset();
}

/*
output:
C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe
[host] ./Cuda.exe Starting...
==10104== NVPROF is profiling process 10104, command: ./Cuda.exe
[host] Using Device 0: NVIDIA GeForce MX450
[host] Data size : 64
[host] Execution Configure (block 64, grid 1)
[host] warmup <<<   1,   64>>>
[host] mathKernel1 <<<   1,   64>>>
[host] mathKernel2 <<<   1,   64>>>
[host] mathKernel3 <<<   1,   64>>>
[host] mathKernel4 <<<   1,   64>>>
==10104== Profiling application: ./Cuda.exe
==10104== Warning: 11 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==10104== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   24.34%  3.2640us         1  3.2640us  3.2640us  3.2640us  warmingup(float*)
                   21.48%  2.8800us         1  2.8800us  2.8800us  2.8800us  mathKernel2(float*)
                   19.57%  2.6240us         1  2.6240us  2.6240us  2.6240us  mathKernel3(float*)
                   18.14%  2.4320us         1  2.4320us  2.4320us  2.4320us  mathKernel1(float*)
                   16.47%  2.2080us         1  2.2080us  2.2080us  2.2080us  mathKernel4(float*)
      API calls:   70.19%  70.862ms         1  70.862ms  70.862ms  70.862ms  cudaSetDevice
                   27.07%  27.330ms         1  27.330ms  27.330ms  27.330ms  cudaDeviceReset
                    1.91%  1.9275ms         5  385.50us  37.500us  1.6709ms  cudaLaunchKernel
                    0.44%  448.40us         1  448.40us  448.40us  448.40us  cudaFree
                    0.15%  153.00us         1  153.00us  153.00us  153.00us  cudaMalloc
                    0.14%  138.10us         5  27.620us  6.0000us  59.600us  cudaDeviceSynchronize
                    0.03%  33.400us       114     292ns       0ns  5.6000us  cuDeviceGetAttribute
                    0.02%  24.800us         2  12.400us     100ns  24.700us  cuDeviceGet
                    0.02%  16.900us         1  16.900us  16.900us  16.900us  cuLibraryUnload
                    0.00%  4.5000us         1  4.5000us  4.5000us  4.5000us  cuDeviceGetLuid
                    0.00%  4.4000us         1  4.4000us  4.4000us  4.4000us  cudaGetDeviceProperties
                    0.00%  2.7000us         1  2.7000us  2.7000us  2.7000us  cuModuleGetLoadingMode
                    0.00%  2.3000us         1  2.3000us  2.3000us  2.3000us  cuDeviceTotalMem
                    0.00%  2.1000us         3     700ns     100ns  1.7000us  cuDeviceGetCount
                    0.00%  1.0000us         1  1.0000us  1.0000us  1.0000us  cuDeviceGetName
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
*/
