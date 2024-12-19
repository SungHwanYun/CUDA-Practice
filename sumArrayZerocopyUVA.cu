#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<string.h>

/*
 * This example demonstrates the use of zero-copy memory to remove the need to
 * explicitly issue a memcpy operation between the host and device. By mapping
 * host, page-locked memory into the device's address space, the address can
 * directly reference a host array and transfer its contents over the PCIe bus.
 *
 * This example compares performing a vector addition with and without zero-copy
 * memory.
 */

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

    return;
}

void initialData(float* ip, int size) {
    for (int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void sumArraysOnHost(float* A, float* B, float* C, const int N) {
    for (int idx = 0; idx < N; idx++) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void sumArrays(float* A, float* B, float* C, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

__global__ void sumArraysZeroCopy(float* A, float* B, float* C, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}
__global__ void sumArraysZeroCopyUVA(float* A, float* B, float* C, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
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
    int ipower = 10;
    if (argc > 1) ipower = atoi(argv[1]);
    int nElem = 1 << ipower;
    size_t nBytes = nElem * sizeof(float);

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
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add vector at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // malloc device global memory
    float* d_A, * d_B, * d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    // set up execution configuration
    int iLen = 512;
    dim3 block(iLen);
    dim3 grid((nElem + block.x - 1) / block.x);
    sumArrays << <grid, block >> > (d_A, d_B, d_C, nElem);

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));

    // free host memory
    free(h_A);
    free(h_B);

    // part 2: using zerocopy memory for array A and B
    // allocate zerocpy memory
    CHECK(cudaHostAlloc((void**)&h_A, nBytes, cudaHostAllocMapped));
    CHECK(cudaHostAlloc((void**)&h_B, nBytes, cudaHostAllocMapped));

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // pass the pointer to device
    CHECK(cudaHostGetDevicePointer((void**)&d_A, (void*)h_A, 0));
    CHECK(cudaHostGetDevicePointer((void**)&d_B, (void*)h_B, 0));

    // add at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // execute kernel with zero copy memory
    sumArraysZeroCopy << <grid, block >> > (d_A, d_B, d_C, nElem);

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free  memory
    CHECK(cudaFreeHost(h_A));
    CHECK(cudaFreeHost(h_B));





    // part 3: using zerocopy UVA memory for array A and B
    // allocate zerocpy UVA memory
    CHECK(cudaHostAlloc((void**)&h_A, nBytes, cudaHostAllocMapped));
    CHECK(cudaHostAlloc((void**)&h_B, nBytes, cudaHostAllocMapped));

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // execute kernel with zero copy memory
    sumArraysZeroCopyUVA << <grid, block >> > (h_A, h_B, d_C, nElem);

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free  memory
    CHECK(cudaFree(d_C));
    CHECK(cudaFreeHost(h_A));
    CHECK(cudaFreeHost(h_B));

    free(hostRef);
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());
}

/*
output:
c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 25
==26600== NVPROF is profiling process 26600, command: ./Cuda.exe 25
[host] Using Device 0: NVIDIA GeForce MX450
[host] Vector size 33554432 power 25  nbytes  128 MB
==26600== Profiling application: ./Cuda.exe 25
==26600== Warning: 5 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==26600== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   33.47%  154.36ms         1  154.36ms  154.36ms  154.36ms  sumArraysZeroCopyUVA(float*, float*, float*, int)
                   27.17%  125.28ms         3  41.760ms  41.438ms  42.190ms  [CUDA memcpy DtoH]
                   19.03%  87.729ms         2  43.865ms  43.855ms  43.874ms  [CUDA memcpy HtoD]
                   18.53%  85.453ms         1  85.453ms  85.453ms  85.453ms  sumArraysZeroCopy(float*, float*, float*, int)
                    1.80%  8.3007ms         1  8.3007ms  8.3007ms  8.3007ms  sumArrays(float*, float*, float*, int)
      API calls:   71.53%  473.17ms         5  94.634ms  43.962ms  196.38ms  cudaMemcpy
                    9.18%  60.699ms         4  15.175ms  13.981ms  16.693ms  cudaHostAlloc
                    8.53%  56.447ms         1  56.447ms  56.447ms  56.447ms  cudaSetDevice
                    6.39%  42.270ms         4  10.567ms  10.374ms  10.681ms  cudaFreeHost
                    3.60%  23.817ms         1  23.817ms  23.817ms  23.817ms  cudaDeviceReset
                    0.40%  2.6233ms         3  874.43us  730.90us  1.1310ms  cudaFree
                    0.19%  1.2300ms         3  410.00us  91.700us  1.0404ms  cudaLaunchKernel
                    0.18%  1.1660ms         3  388.67us  199.00us  615.60us  cudaMalloc
                    0.00%  26.200us         2  13.100us  1.6000us  24.600us  cudaHostGetDevicePointer
                    0.00%  24.800us       114     217ns       0ns  3.6000us  cuDeviceGetAttribute
                    0.00%  14.800us         1  14.800us  14.800us  14.800us  cuLibraryUnload
                    0.00%  4.5000us         1  4.5000us  4.5000us  4.5000us  cudaGetDeviceProperties
                    0.00%  3.1000us         1  3.1000us  3.1000us  3.1000us  cuDeviceTotalMem
                    0.00%  2.6000us         1  2.6000us  2.6000us  2.6000us  cuModuleGetLoadingMode
                    0.00%  2.5000us         3     833ns     100ns  2.1000us  cuDeviceGetCount
                    0.00%     800ns         2     400ns     100ns     700ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid
*/
