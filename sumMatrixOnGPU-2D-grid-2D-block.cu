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

// Initialize the data pointed to by the pointer ip.
void initialInt(int* ip, const int N) {
    for (int i = 0; i < N; i++) {
        ip[i] = i;
    }
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

void sumMatrixOnHost(float* A, float* B, float* C, const int nx, const int ny) {
    float* ia = A, * ib = B, * ic = C;
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx; ib += nx; ic += nx;
    }
}

// Initialize the data pointed to by the pointer ip.
#include<time.h>
void initialData(float* ip, const int N) {
    // generate different seed for random number
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < N; i++) {
        ip[i] = (float)(rand() & 0xff) / 10.0f;
    }
}

__global__ void sumMatrixOnGPU2D2D(float* A, float* B, float* C, const int nx, const int ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * nx + ix;
    if (ix < nx && iy < ny) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char **argv) {
    printf("[host] %s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("[host] Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of matrix
    int nx = 1 << 14;
    int ny = 1 << 14;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("[host] Matrix size : nx %d, ny %d\n", nx, ny);

    // malloc host memory
    float* h_A, * h_B, * hostRef, * gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    // initialize data at host side
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);

    // malloc device global memory
    float* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, nBytes);
    cudaMalloc((void**)&d_B, nBytes);
    cudaMalloc((void**)&d_C, nBytes);

    // transfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // set up execution configuration
    int dimx = 32, dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // invoke the kernel
    sumMatrixOnGPU2D2D << <grid, block >> > (d_A, d_B, d_C, nx, ny);
    cudaDeviceSynchronize();

    // copy kernel result back to host side
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    // check device results
    checkResult(hostRef, gpuRef, nxy);
    
    // free host and device memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(hostRef); free(gpuRef);

    // reset device
    cudaDeviceReset();
}

/*
output:
C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe
[host] ./Cuda.exe Starting...
==23544== NVPROF is profiling process 23544, command: ./Cuda.exe
[host] Using Device 0: NVIDIA GeForce MX450
[host] Matrix size : nx 16384, ny 16384
[host] Arrays match.

==23544== Profiling application: ./Cuda.exe
==23544== Warning: 8 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==23544== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   50.08%  736.63ms         2  368.31ms  361.52ms  375.11ms  [CUDA memcpy HtoD]
                   26.69%  392.65ms         1  392.65ms  392.65ms  392.65ms  [CUDA memcpy DtoH]
                   23.23%  341.62ms         1  341.62ms  341.62ms  341.62ms  sumMatrixOnGPU2D2D(float*, float*, float*, int, int)
      API calls:   61.48%  1.63047s         3  543.49ms  361.54ms  875.81ms  cudaMemcpy
                   15.34%  406.67ms         3  135.56ms  1.9769ms  389.35ms  cudaMalloc
                   12.89%  341.71ms         1  341.71ms  341.71ms  341.71ms  cudaDeviceSynchronize
                    5.61%  148.75ms         3  49.585ms  11.103ms  117.34ms  cudaFree
                    2.85%  75.508ms         1  75.508ms  75.508ms  75.508ms  cudaSetDevice
                    1.43%  38.019ms         1  38.019ms  38.019ms  38.019ms  cudaDeviceReset
                    0.40%  10.541ms         1  10.541ms  10.541ms  10.541ms  cudaLaunchKernel
                    0.01%  136.40us         1  136.40us  136.40us  136.40us  cuModuleGetLoadingMode
                    0.00%  25.600us       114     224ns       0ns  3.9000us  cuDeviceGetAttribute
                    0.00%  10.500us         1  10.500us  10.500us  10.500us  cuLibraryUnload
                    0.00%  3.7000us         1  3.7000us  3.7000us  3.7000us  cudaGetDeviceProperties
                    0.00%  2.3000us         3     766ns       0ns  1.7000us  cuDeviceGetCount
                    0.00%  2.3000us         1  2.3000us  2.3000us  2.3000us  cuDeviceTotalMem
                    0.00%  1.0000us         2     500ns     200ns     800ns  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid
*/
