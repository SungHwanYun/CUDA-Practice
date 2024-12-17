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
__global__ void sumMatrixOnGPU1D1D(float* A, float* B, float* C, const int nx, const int ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < nx) {
        for (int iy = 0; iy < ny; iy++) {
            unsigned int idx = iy * nx + ix;
            C[idx] = A[idx] + B[idx];
        }
    }
}
__global__ void sumMatrixOnGPU2D1D(float* A, float* B, float* C, const int nx, const int ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y;
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
    int dimx = 32;
    dim3 block(dimx, 1);
    dim3 grid((nx + block.x - 1) / block.x, ny);

    // invoke the kernel
    sumMatrixOnGPU2D1D << <grid, block >> > (d_A, d_B, d_C, nx, ny);
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
==11636== NVPROF is profiling process 11636, command: ./Cuda.exe
[host] Using Device 0: NVIDIA GeForce MX450
[host] Matrix size : nx 16384, ny 16384
[host] Arrays match.

==11636== Profiling application: ./Cuda.exe
==11636== Warning: 37 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==11636== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   49.38%  739.83ms         2  369.92ms  363.57ms  376.26ms  [CUDA memcpy HtoD]
                   27.81%  416.65ms         1  416.65ms  416.65ms  416.65ms  [CUDA memcpy DtoH]
                   22.81%  341.81ms         1  341.81ms  341.81ms  341.81ms  sumMatrixOnGPU2D1D(float*, float*, float*, int, int)
      API calls:   59.22%  1.43075s         3  476.92ms  363.61ms  650.09ms  cudaMemcpy
                   15.06%  363.82ms         3  121.27ms  2.2682ms  341.71ms  cudaMalloc
                   14.15%  341.86ms         1  341.86ms  341.86ms  341.86ms  cudaDeviceSynchronize
                    6.85%  165.54ms         3  55.181ms  10.570ms  134.70ms  cudaFree
                    3.08%  74.413ms         1  74.413ms  74.413ms  74.413ms  cudaSetDevice
                    1.58%  38.265ms         1  38.265ms  38.265ms  38.265ms  cudaDeviceReset
                    0.06%  1.3330ms         1  1.3330ms  1.3330ms  1.3330ms  cudaLaunchKernel
                    0.00%  19.200us       114     168ns       0ns  2.8000us  cuDeviceGetAttribute
                    0.00%  15.600us         1  15.600us  15.600us  15.600us  cuLibraryUnload
                    0.00%  4.9000us         2  2.4500us       0ns  4.9000us  cuDeviceGet
                    0.00%  3.5000us         1  3.5000us  3.5000us  3.5000us  cudaGetDeviceProperties
                    0.00%  2.3000us         3     766ns     100ns  1.9000us  cuDeviceGetCount
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuModuleGetLoadingMode
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuDeviceTotalMem
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid
*/
