#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<time.h>
#include<string.h>

// If your development environment is Linux, remove the comment below.
// #define LINUX

// error check
#define CHECK(call) \
{ \
    const cudaError_t error = call;\
    if (error != cudaSuccess) { \
        printf("[device] Error: %s %d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// Check if the computation results of the CPU and GPU are the same
void checkResult(float* hostRef, float* gpuRef, const int N) {
    double epsilon = 1.0e-8;
    bool match = 1;
    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("[host] Arrays do not match!\n");
            printf("[host] host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match) printf("[host] Arrays match.\n\n");
}

// Initialize the data pointed to by the pointer ip.
void initialData(float* ip, const int N) {
    // generate different seed for random number
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < N; i++) {
        ip[i] = (float)(rand() & 0xff) / 10.0f;
    }
}

// C[i] = A[i] + B[i], 0<= i <= N-1
void sumArraysOnHost(float* A, float* B, float* C, const int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C, int const N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
    //printf("[device] thread %d : %5.2f + %5.2f = %5.2f\n", i, A[i], B[i], C[i]);
}

double cpuSecond() {
#ifdef LINUX
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tv.tv_sec + (double)tv.tv_usec * 1e-6);
#else 
    return 0.0f;
#endif
}

int main(int argc, char **argv) {
    printf("[host] %s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("[host] Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of vectors
    int nElem = 1 << 24;
    printf("[host] Vector size %d\n", nElem);

    // malloc host memory
    size_t nBytes = nElem * sizeof(float);
    float* h_A, * h_B, * hostRef, * gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    double iStart, iElaps;

    // initialize data at host side
    iStart = cpuSecond();
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    iElaps = cpuSecond() - iStart;
    memset(hostRef, 0, sizeof(hostRef));
    memset(gpuRef, 0, sizeof(gpuRef));

    // add vector at host side for result checks
    iStart = cpuSecond();
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    iElaps = cpuSecond() - iStart;

    // malloc device global memory
    float* d_A, * d_B, * d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    // transfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // invoke kernel at host side
    int iLen = 1 << 10;
    dim3 block(iLen, 1, 1);
    dim3 grid((nElem + block.x - 1) / block.x, 1, 1);

    iStart = cpuSecond();
    sumArraysOnGPU << <grid, block >> > (d_A, d_B, d_C, nElem);
    iElaps = cpuSecond() - iStart;
    printf("[host] sumArraysOnGPU <<<%d, %d>>> Time elapsed %f sec\n", 
        grid.x, block.x, iElaps);

    // copy kernel result back to host side
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
}

/*
output:
C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe
[host] ./Cuda.exe Starting...
==20084== NVPROF is profiling process 20084, command: ./Cuda.exe
[host] Using Device 0: NVIDIA GeForce MX450
[host] Vector size 16777216
[host] sumArraysOnGPU <<<16384, 1024>>> Time elapsed 0.000000 sec
[host] Arrays match.

==20084== Profiling application: ./Cuda.exe
==20084== Warning: 36 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==20084== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   62.71%  43.880ms         2  21.940ms  21.776ms  22.104ms  [CUDA memcpy HtoD]
                   31.41%  21.979ms         1  21.979ms  21.979ms  21.979ms  [CUDA memcpy DtoH]
                    5.88%  4.1108ms         1  4.1108ms  4.1108ms  4.1108ms  sumArraysOnGPU(float*, float*, float*, int)
      API calls:   38.87%  138.87ms         1  138.87ms  138.87ms  138.87ms  cudaSetDevice
                   26.96%  96.324ms         1  96.324ms  96.324ms  96.324ms  cudaLaunchKernel
                   20.71%  73.983ms         3  24.661ms  22.211ms  26.619ms  cudaMemcpy
                    9.82%  35.090ms         1  35.090ms  35.090ms  35.090ms  cuDevicePrimaryCtxRelease
                    2.95%  10.536ms         3  3.5121ms  166.00us  9.6766ms  cudaMalloc
                    0.56%  1.9937ms         3  664.57us  398.20us  1.1074ms  cudaFree
                    0.06%  217.50us         1  217.50us  217.50us  217.50us  cuModuleGetLoadingMode
                    0.06%  211.40us       114  1.8540us       0ns  196.00us  cuDeviceGetAttribute
                    0.01%  46.400us         1  46.400us  46.400us  46.400us  cuLibraryUnload
                    0.00%  5.9000us         1  5.9000us  5.9000us  5.9000us  cudaGetDeviceProperties
                    0.00%  2.4000us         3     800ns     100ns  2.0000us  cuDeviceGetCount
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuDeviceTotalMem
                    0.00%  1.1000us         2     550ns     100ns  1.0000us  cuDeviceGet
                    0.00%  1.0000us         1  1.0000us  1.0000us  1.0000us  cuDeviceGetName
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceGetLuid
*/
