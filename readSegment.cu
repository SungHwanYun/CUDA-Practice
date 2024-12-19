#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<string.h>

#define CHECK(call) \
{ \
    const cudaError_t error = call;\
    if (error != cudaSuccess) { \
        printf("[device] Error: %s %d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(-10*error); \
    } \
}

void checkResult(float* hostRef, float* gpuRef, const int N) {
    double epsilon = 1.0E-8;
    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            printf("[host] Arrays do not match!\n");
            printf("[host] host %5.2f gpu %5.2f at current %d\n", hostRef[i],
                gpuRef[i], i);
            break;
        }
    }
}

void initialData(float* ip, int size) {
    for (int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void sumArraysOnHost(float* A, float* B, float* C, const int n, int offset) {
    for (int i = offset, k = 0; i < n; i++, k++) {
        C[k] = A[i] + B[i];
    }
}
__global__ void warmup(float* A, float* B, float* C, const int n, int offset) {
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = k + offset;
    if (i < n) C[k] = A[i] + B[i];
}
__global__ void readOffset(float* A, float* B, float* C, const int n, int offset) {
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = k + offset;
    if (i < n) C[k] = A[i] + B[i];
}
int main(int argc, char** argv) {
    // set up device
    int dev = 0;
    CHECK(cudaSetDevice(dev));

    // get device properties
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    // check if support mapped memory
    if (!deviceProp.canMapHostMemory) {
        printf("[host] Device %d does not support mapping CPU host memory!\n", dev);
        CHECK(cudaDeviceReset());
        exit(EXIT_SUCCESS);
    }

    printf("[host] Using Device %d: %s\n", dev, deviceProp.name);

    // set up data size of vectors
    int ipower = 25;
    int nElem = 1 << ipower;
    size_t nBytes = nElem * sizeof(float);

    // set up execution configuration
    int blocksize = 512;
    int offset = 0;
    if (argc > 1) offset = atoi(argv[1]);
    if (argc > 2) blocksize = atoi(argv[2]);
    dim3 block(blocksize, 1);
    dim3 grid((nElem + block.x - 1) / block.x, 1);
    printf("[host] offset : %d, blocksize : %d\n", offset, blocksize);

    if (ipower < 18) {
        printf("[host] Vector size %d power %d  nbytes  %3.0f KB\n", nElem, ipower,
            (float)nBytes / (1024.0f));
    } else {
        printf("[host] Vector size %d power %d  nbytes  %3.0f MB\n", nElem, ipower,
            (float)nBytes / (1024.0f * 1024.0f));
    }

    // part 1: using device memory
    // malloc host memory
    float* h_A, * h_B, * hostRef, * gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    // initialize data at host side
    initialData(h_A, nElem);
    memcpy(h_B, h_A, nBytes);;

    // add vector at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem, offset);

    // malloc device global memory
    float* d_A, * d_B, * d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    // kernel1:
    warmup << <grid, block >> > (d_A, d_B, d_C, nElem, offset);
    cudaDeviceSynchronize();

    // kernel 2 : readOffset
    readOffset << <grid, block >> > (d_A, d_B, d_C, nElem, offset);
    cudaDeviceSynchronize();

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // check device results
    checkResult(hostRef, gpuRef, nElem - offset);

    // free device global memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());
}

/*
output:
c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe
==16792== NVPROF is profiling process 16792, command: ./Cuda.exe
[host] Using Device 0: NVIDIA GeForce MX450
[host] offset : 0, blocksize : 512
[host] Vector size 33554432 power 25  nbytes  128 MB
==16792== Profiling application: ./Cuda.exe
==16792== Warning: 29 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==16792== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   59.80%  87.470ms         2  43.735ms  43.716ms  43.755ms  [CUDA memcpy HtoD]
                   28.93%  42.317ms         1  42.317ms  42.317ms  42.317ms  [CUDA memcpy DtoH]
                    5.64%  8.2436ms         1  8.2436ms  8.2436ms  8.2436ms  readOffset(float*, float*, float*, int, int)
                    5.63%  8.2413ms         1  8.2413ms  8.2413ms  8.2413ms  warmup(float*, float*, float*, int, int)
      API calls:   54.32%  140.06ms         3  46.685ms  42.746ms  53.526ms  cudaMemcpy
                   26.01%  67.072ms         1  67.072ms  67.072ms  67.072ms  cudaSetDevice
                   10.55%  27.199ms         1  27.199ms  27.199ms  27.199ms  cudaDeviceReset
                    6.41%  16.536ms         2  8.2680ms  8.2640ms  8.2721ms  cudaDeviceSynchronize
                    1.36%  3.5169ms         3  1.1723ms  834.30us  1.5794ms  cudaFree
                    0.84%  2.1593ms         3  719.77us  518.10us  961.60us  cudaMalloc
                    0.48%  1.2420ms         2  621.00us  13.900us  1.2281ms  cudaLaunchKernel
                    0.01%  30.000us         1  30.000us  30.000us  30.000us  cuLibraryUnload
                    0.01%  18.800us       114     164ns       0ns  2.6000us  cuDeviceGetAttribute
                    0.00%  4.0000us         1  4.0000us  4.0000us  4.0000us  cudaGetDeviceProperties
                    0.00%  2.6000us         3     866ns     100ns  2.2000us  cuDeviceGetCount
                    0.00%  2.2000us         1  2.2000us  2.2000us  2.2000us  cuDeviceTotalMem
                    0.00%  2.2000us         1  2.2000us  2.2000us  2.2000us  cuDeviceGetLuid
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuModuleGetLoadingMode
                    0.00%  1.7000us         2     850ns     100ns  1.6000us  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 11
==8268== NVPROF is profiling process 8268, command: ./Cuda.exe 11
[host] Using Device 0: NVIDIA GeForce MX450
[host] offset : 11, blocksize : 512
[host] Vector size 33554432 power 25  nbytes  128 MB
==8268== Profiling application: ./Cuda.exe 11
==8268== Warning: 29 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==8268== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   59.23%  87.428ms         2  43.714ms  43.604ms  43.824ms  [CUDA memcpy HtoD]
                   29.69%  43.831ms         1  43.831ms  43.831ms  43.831ms  [CUDA memcpy DtoH]
                    5.54%  8.1758ms         1  8.1758ms  8.1758ms  8.1758ms  readOffset(float*, float*, float*, int, int)
                    5.54%  8.1731ms         1  8.1731ms  8.1731ms  8.1731ms  warmup(float*, float*, float*, int, int)
      API calls:   53.74%  142.75ms         3  47.583ms  43.922ms  54.606ms  cudaMemcpy
                   25.14%  66.784ms         1  66.784ms  66.784ms  66.784ms  cudaSetDevice
                   12.54%  33.320ms         1  33.320ms  33.320ms  33.320ms  cudaDeviceReset
                    6.18%  16.415ms         2  8.2075ms  8.1765ms  8.2384ms  cudaDeviceSynchronize
                    1.54%  4.0938ms         3  1.3646ms  1.1333ms  1.4886ms  cudaFree
                    0.46%  1.2235ms         2  611.75us  21.600us  1.2019ms  cudaLaunchKernel
                    0.37%  985.10us         3  328.37us  245.50us  408.30us  cudaMalloc
                    0.01%  18.800us       114     164ns       0ns  3.1000us  cuDeviceGetAttribute
                    0.01%  18.000us         1  18.000us  18.000us  18.000us  cuLibraryUnload
                    0.00%  4.1000us         1  4.1000us  4.1000us  4.1000us  cudaGetDeviceProperties
                    0.00%  3.2000us         3  1.0660us     100ns  2.8000us  cuDeviceGetCount
                    0.00%  2.2000us         1  2.2000us  2.2000us  2.2000us  cuDeviceTotalMem
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuModuleGetLoadingMode
                    0.00%     800ns         2     400ns     100ns     700ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 128
==15580== NVPROF is profiling process 15580, command: ./Cuda.exe 128
[host] Using Device 0: NVIDIA GeForce MX450
[host] offset : 128, blocksize : 512
[host] Vector size 33554432 power 25  nbytes  128 MB
==15580== Profiling application: ./Cuda.exe 128
==15580== Warning: 34 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==15580== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   58.40%  87.624ms         2  43.812ms  43.772ms  43.852ms  [CUDA memcpy HtoD]
                   30.61%  45.937ms         1  45.937ms  45.937ms  45.937ms  [CUDA memcpy DtoH]
                    5.49%  8.2440ms         1  8.2440ms  8.2440ms  8.2440ms  readOffset(float*, float*, float*, int, int)
                    5.49%  8.2439ms         1  8.2439ms  8.2439ms  8.2439ms  warmup(float*, float*, float*, int, int)
      API calls:   57.16%  145.14ms         3  48.381ms  43.940ms  54.776ms  cudaMemcpy
                   25.03%  63.561ms         1  63.561ms  63.561ms  63.561ms  cudaSetDevice
                    9.41%  23.888ms         1  23.888ms  23.888ms  23.888ms  cudaDeviceReset
                    6.56%  16.656ms         2  8.3282ms  8.3071ms  8.3494ms  cudaDeviceSynchronize
                    0.87%  2.2020ms         3  734.00us  526.50us  965.00us  cudaFree
                    0.48%  1.2138ms         3  404.60us  323.50us  448.60us  cudaMalloc
                    0.47%  1.2034ms         2  601.70us  144.90us  1.0585ms  cudaLaunchKernel
                    0.01%  18.200us       114     159ns       0ns  2.4000us  cuDeviceGetAttribute
                    0.01%  15.300us         1  15.300us  15.300us  15.300us  cuLibraryUnload
                    0.00%  4.2000us         1  4.2000us  4.2000us  4.2000us  cudaGetDeviceProperties
                    0.00%  2.1000us         3     700ns     100ns  1.7000us  cuDeviceGetCount
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuModuleGetLoadingMode
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuDeviceTotalMem
                    0.00%     900ns         2     450ns     100ns     800ns  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
*/
