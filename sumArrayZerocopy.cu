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
c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe
==16532== NVPROF is profiling process 16532, command: ./Cuda.exe
[host] Using Device 0: NVIDIA GeForce MX450
[host] Vector size 1024 power 10  nbytes    4 KB
==16532== Profiling application: ./Cuda.exe
==16532== Warning: 29 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==16532== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   42.84%  8.8960us         1  8.8960us  8.8960us  8.8960us  sumArraysZeroCopy(float*, float*, float*, int)
                   23.27%  4.8320us         2  2.4160us  1.9840us  2.8480us  [CUDA memcpy DtoH]
                   17.41%  3.6160us         2  1.8080us  1.3760us  2.2400us  [CUDA memcpy HtoD]
                   16.49%  3.4240us         1  3.4240us  3.4240us  3.4240us  sumArrays(float*, float*, float*, int)
      API calls:   77.91%  103.75ms         1  103.75ms  103.75ms  103.75ms  cudaSetDevice
                   19.41%  25.853ms         1  25.853ms  25.853ms  25.853ms  cudaDeviceReset
                    1.15%  1.5305ms         2  765.25us  38.000us  1.4925ms  cudaLaunchKernel
                    0.82%  1.0888ms         2  544.40us  25.800us  1.0630ms  cudaHostAlloc
                    0.27%  360.50us         4  90.125us  41.400us  161.00us  cudaMemcpy
                    0.16%  212.80us         2  106.40us  11.000us  201.80us  cudaFreeHost
                    0.14%  188.60us         3  62.866us  2.0000us  182.90us  cudaMalloc
                    0.10%  128.80us         3  42.933us  1.8000us  119.80us  cudaFree
                    0.02%  20.100us       114     176ns       0ns  2.9000us  cuDeviceGetAttribute
                    0.01%  14.000us         1  14.000us  14.000us  14.000us  cuLibraryUnload
                    0.00%  5.8000us         1  5.8000us  5.8000us  5.8000us  cudaGetDeviceProperties
                    0.00%  3.9000us         2  1.9500us     400ns  3.5000us  cudaHostGetDevicePointer
                    0.00%  2.0000us         3     666ns     100ns  1.6000us  cuDeviceGetCount
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuDeviceTotalMem
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuModuleGetLoadingMode
                    0.00%     800ns         2     400ns     100ns     700ns  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 20
==620== NVPROF is profiling process 620, command: ./Cuda.exe 20
[host] Using Device 0: NVIDIA GeForce MX450
[host] Vector size 1048576 power 20  nbytes    4 MB
==620== Profiling application: ./Cuda.exe 20
==620== Warning: 26 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==620== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   33.58%  2.7792ms         2  1.3896ms  1.3654ms  1.4138ms  [CUDA memcpy HtoD]
                   32.29%  2.6728ms         1  2.6728ms  2.6728ms  2.6728ms  sumArraysZeroCopy(float*, float*, float*, int)
                   31.03%  2.5687ms         2  1.2844ms  1.2841ms  1.2847ms  [CUDA memcpy DtoH]
                    3.10%  256.61us         1  256.61us  256.61us  256.61us  sumArrays(float*, float*, float*, int)
      API calls:   64.77%  66.353ms         1  66.353ms  66.353ms  66.353ms  cudaSetDevice
                   21.88%  22.419ms         1  22.419ms  22.419ms  22.419ms  cudaDeviceReset
                    8.85%  9.0666ms         4  2.2667ms  1.1850ms  4.4371ms  cudaMemcpy
                    1.54%  1.5750ms         2  787.50us  738.00us  837.00us  cudaHostAlloc
                    1.31%  1.3409ms         2  670.45us  88.900us  1.2520ms  cudaLaunchKernel
                    0.57%  586.10us         2  293.05us  279.80us  306.30us  cudaFreeHost
                    0.55%  564.10us         3  188.03us  95.800us  240.90us  cudaFree
                    0.46%  471.90us         3  157.30us  88.300us  292.00us  cudaMalloc
                    0.02%  20.500us       114     179ns       0ns  2.9000us  cuDeviceGetAttribute
                    0.02%  20.400us         2  10.200us     600ns  19.800us  cudaHostGetDevicePointer
                    0.01%  11.800us         1  11.800us  11.800us  11.800us  cuLibraryUnload
                    0.00%  5.1000us         1  5.1000us  5.1000us  5.1000us  cudaGetDeviceProperties
                    0.00%  2.8000us         1  2.8000us  2.8000us  2.8000us  cuModuleGetLoadingMode
                    0.00%  2.5000us         3     833ns     100ns  1.9000us  cuDeviceGetCount
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuDeviceTotalMem
                    0.00%  1.4000us         2     700ns     100ns  1.3000us  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 25
==14596== NVPROF is profiling process 14596, command: ./Cuda.exe 25
[host] Using Device 0: NVIDIA GeForce MX450
[host] Vector size 33554432 power 25  nbytes  128 MB
==14596== Profiling application: ./Cuda.exe 25
==14596== Warning: 31 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==14596== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   33.35%  88.772ms         2  44.386ms  44.010ms  44.762ms  [CUDA memcpy HtoD]
                   32.25%  85.854ms         1  85.854ms  85.854ms  85.854ms  sumArraysZeroCopy(float*, float*, float*, int)
                   31.31%  83.340ms         2  41.670ms  41.485ms  41.855ms  [CUDA memcpy DtoH]
                    3.10%  8.2427ms         1  8.2427ms  8.2427ms  8.2427ms  sumArrays(float*, float*, float*, int)
      API calls:   62.97%  277.61ms         4  69.402ms  44.845ms  127.88ms  cudaMemcpy
                   13.67%  60.258ms         1  60.258ms  60.258ms  60.258ms  cudaSetDevice
                   10.92%  48.161ms         2  24.080ms  23.542ms  24.619ms  cudaHostAlloc
                    5.71%  25.153ms         2  12.577ms  11.300ms  13.854ms  cudaFreeHost
                    5.45%  24.049ms         1  24.049ms  24.049ms  24.049ms  cudaDeviceReset
                    0.72%  3.1961ms         3  1.0654ms  799.60us  1.3533ms  cudaFree
                    0.29%  1.2990ms         3  433.00us  245.50us  650.10us  cudaMalloc
                    0.24%  1.0724ms         2  536.20us  87.400us  985.00us  cudaLaunchKernel
                    0.00%  19.300us       114     169ns       0ns  2.4000us  cuDeviceGetAttribute
                    0.00%  18.900us         2  9.4500us     600ns  18.300us  cudaHostGetDevicePointer
                    0.00%  18.200us         1  18.200us  18.200us  18.200us  cuLibraryUnload
                    0.00%  5.4000us         1  5.4000us  5.4000us  5.4000us  cudaGetDeviceProperties
                    0.00%  3.1000us         3  1.0330us     100ns  2.6000us  cuDeviceGetCount
                    0.00%  2.5000us         1  2.5000us  2.5000us  2.5000us  cuModuleGetLoadingMode
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuDeviceTotalMem
                    0.00%  1.2000us         2     600ns     200ns  1.0000us  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 30
==6504== NVPROF is profiling process 6504, command: ./Cuda.exe 30
[host] Using Device 0: NVIDIA GeForce MX450
[host] Vector size 1073741824 power 30  nbytes  4096 MB
[device] Error: C:\coding\Cuda\Cuda\main.cu 115, code:2, reason: out of memory
==6504== Profiling application: ./Cuda.exe 30
==6504== Warning: 34 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==6504== Profiling result:
No kernels were profiled.
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
      API calls:   91.96%  17.8817s         3  5.96055s  572.98ms  16.6002s  cudaMalloc
                    7.72%  1.50173s         1  1.50173s  1.50173s  1.50173s  cuDevicePrimaryCtxRelease
                    0.31%  61.098ms         1  61.098ms  61.098ms  61.098ms  cudaSetDevice
                    0.00%  252.30us         1  252.30us  252.30us  252.30us  cudaGetErrorString
                    0.00%  18.500us       114     162ns       0ns  3.2000us  cuDeviceGetAttribute
                    0.00%  5.2000us         1  5.2000us  5.2000us  5.2000us  cudaGetDeviceProperties
                    0.00%  2.2000us         3     733ns     100ns  1.9000us  cuDeviceGetCount
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuDeviceTotalMem
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuModuleGetLoadingMode
                    0.00%  1.5000us         2     750ns     100ns  1.4000us  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
======== Error: Application returned non-zero code -20
*/
