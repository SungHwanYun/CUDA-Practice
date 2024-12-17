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

int main(int argc, char** argv) {
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
    if (argc > 2) {
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
    }
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    printf("[host] Execution configure : grid(%d, %d), block(%d, %d)\n", 
        grid.x, grid.y, block.x, block.y);

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
C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 32 32
[host] ./Cuda.exe Starting...
==7368== NVPROF is profiling process 7368, command: ./Cuda.exe 32 32
[host] Using Device 0: NVIDIA GeForce MX450
[host] Matrix size : nx 16384, ny 16384
[host] Execution configure : grid(512, 512), block(32, 32)
[host] Arrays match.

==7368== Profiling application: ./Cuda.exe 32 32
==7368== Warning: 32 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==7368== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   49.43%  719.87ms         2  359.94ms  354.49ms  365.38ms  [CUDA memcpy HtoD]
                   27.18%  395.79ms         1  395.79ms  395.79ms  395.79ms  [CUDA memcpy DtoH]
                   23.39%  340.60ms         1  340.60ms  340.60ms  340.60ms  sumMatrixOnGPU2D2D(float*, float*, float*, int, int)
      API calls:   58.80%  1.40533s         3  468.44ms  365.44ms  643.62ms  cudaMemcpy
                   16.73%  399.93ms         3  133.31ms  1.9281ms  380.77ms  cudaMalloc
                   14.25%  340.64ms         1  340.64ms  340.64ms  340.64ms  cudaDeviceSynchronize
                    5.74%  137.25ms         3  45.751ms  11.290ms  106.53ms  cudaFree
                    3.10%  74.176ms         1  74.176ms  74.176ms  74.176ms  cudaSetDevice
                    1.29%  30.876ms         1  30.876ms  30.876ms  30.876ms  cudaDeviceReset
                    0.08%  1.9524ms         1  1.9524ms  1.9524ms  1.9524ms  cudaLaunchKernel
                    0.00%  22.800us         1  22.800us  22.800us  22.800us  cuLibraryUnload
                    0.00%  19.400us       114     170ns       0ns  3.1000us  cuDeviceGetAttribute
                    0.00%  9.3000us         1  9.3000us  9.3000us  9.3000us  cudaGetDeviceProperties
                    0.00%  2.8000us         3     933ns     200ns  2.3000us  cuDeviceGetCount
                    0.00%  2.2000us         1  2.2000us  2.2000us  2.2000us  cuDeviceTotalMem
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuModuleGetLoadingMode
                    0.00%  1.0000us         2     500ns     100ns     900ns  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 32 16
[host] ./Cuda.exe Starting...
==10600== NVPROF is profiling process 10600, command: ./Cuda.exe 32 16
[host] Using Device 0: NVIDIA GeForce MX450
[host] Matrix size : nx 16384, ny 16384
[host] Execution configure : grid(512, 1024), block(32, 16)
[host] Arrays match.

==10600== Profiling application: ./Cuda.exe 32 16
==10600== Warning: 34 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==10600== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   49.43%  722.95ms         2  361.48ms  355.17ms  367.78ms  [CUDA memcpy HtoD]
                   27.22%  398.06ms         1  398.06ms  398.06ms  398.06ms  [CUDA memcpy DtoH]
                   23.35%  341.52ms         1  341.52ms  341.52ms  341.52ms  sumMatrixOnGPU2D2D(float*, float*, float*, int, int)
      API calls:   58.22%  1.33310s         3  444.37ms  367.80ms  566.85ms  cudaMemcpy
                   16.39%  375.30ms         3  125.10ms  2.2336ms  355.83ms  cudaMalloc
                   14.92%  341.55ms         1  341.55ms  341.55ms  341.55ms  cudaDeviceSynchronize
                    5.79%  132.49ms         3  44.162ms  9.4659ms  106.09ms  cudaFree
                    3.33%  76.265ms         1  76.265ms  76.265ms  76.265ms  cudaSetDevice
                    1.28%  29.316ms         1  29.316ms  29.316ms  29.316ms  cudaDeviceReset
                    0.07%  1.6695ms         1  1.6695ms  1.6695ms  1.6695ms  cudaLaunchKernel
                    0.00%  18.600us       114     163ns       0ns  3.5000us  cuDeviceGetAttribute
                    0.00%  12.500us         1  12.500us  12.500us  12.500us  cuLibraryUnload
                    0.00%  4.4000us         1  4.4000us  4.4000us  4.4000us  cudaGetDeviceProperties
                    0.00%  2.6000us         3     866ns     100ns  2.2000us  cuDeviceGetCount
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuDeviceTotalMem
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuModuleGetLoadingMode
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     700ns         2     350ns       0ns     700ns  cuDeviceGet
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 16 32
[host] ./Cuda.exe Starting...
==21208== NVPROF is profiling process 21208, command: ./Cuda.exe 16 32
[host] Using Device 0: NVIDIA GeForce MX450
[host] Matrix size : nx 16384, ny 16384
[host] Execution configure : grid(1024, 512), block(16, 32)
[host] Arrays match.

==21208== Profiling application: ./Cuda.exe 16 32
==21208== Warning: 37 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==21208== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   47.61%  725.46ms         2  362.73ms  360.80ms  364.66ms  [CUDA memcpy HtoD]
                   26.47%  403.38ms         1  403.38ms  403.38ms  403.38ms  sumMatrixOnGPU2D2D(float*, float*, float*, int, int)
                   25.92%  394.97ms         1  394.97ms  394.97ms  394.97ms  [CUDA memcpy DtoH]
      API calls:   57.90%  1.49415s         3  498.05ms  364.77ms  733.99ms  cudaMemcpy
                   15.74%  406.24ms         3  135.41ms  2.0368ms  389.67ms  cudaMalloc
                   15.63%  403.40ms         1  403.40ms  403.40ms  403.40ms  cudaDeviceSynchronize
                    6.48%  167.32ms         3  55.774ms  8.4694ms  139.71ms  cudaFree
                    3.14%  81.131ms         1  81.131ms  81.131ms  81.131ms  cudaSetDevice
                    1.03%  26.602ms         1  26.602ms  26.602ms  26.602ms  cudaDeviceReset
                    0.07%  1.7240ms         1  1.7240ms  1.7240ms  1.7240ms  cudaLaunchKernel
                    0.00%  55.200us       114     484ns       0ns  17.500us  cuDeviceGetAttribute
                    0.00%  10.400us         1  10.400us  10.400us  10.400us  cuLibraryUnload
                    0.00%  3.6000us         1  3.6000us  3.6000us  3.6000us  cudaGetDeviceProperties
                    0.00%  2.1000us         3     700ns       0ns  1.8000us  cuDeviceGetCount
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuDeviceTotalMem
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuModuleGetLoadingMode
                    0.00%  1.0000us         1  1.0000us  1.0000us  1.0000us  cuDeviceGetName
                    0.00%     800ns         2     400ns       0ns     800ns  cuDeviceGet
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 16 16
[host] ./Cuda.exe Starting...
==21160== NVPROF is profiling process 21160, command: ./Cuda.exe 16 16
[host] Using Device 0: NVIDIA GeForce MX450
[host] Matrix size : nx 16384, ny 16384
[host] Execution configure : grid(1024, 1024), block(16, 16)
[host] Arrays match.

==21160== Profiling application: ./Cuda.exe 16 16
==21160== Warning: 18 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==21160== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   47.49%  717.34ms         2  358.67ms  351.48ms  365.86ms  [CUDA memcpy HtoD]
                   26.38%  398.44ms         1  398.44ms  398.44ms  398.44ms  sumMatrixOnGPU2D2D(float*, float*, float*, int, int)
                   26.13%  394.65ms         1  394.65ms  394.65ms  394.65ms  [CUDA memcpy DtoH]
      API calls:   56.94%  1.32491s         3  441.64ms  365.89ms  563.89ms  cudaMemcpy
                   17.12%  398.46ms         1  398.46ms  398.46ms  398.46ms  cudaDeviceSynchronize
                   14.98%  348.55ms         3  116.18ms  2.1019ms  332.20ms  cudaMalloc
                    6.51%  151.47ms         3  50.489ms  9.1992ms  123.67ms  cudaFree
                    3.23%  75.065ms         1  75.065ms  75.065ms  75.065ms  cudaSetDevice
                    1.16%  26.934ms         1  26.934ms  26.934ms  26.934ms  cudaDeviceReset
                    0.07%  1.6189ms         1  1.6189ms  1.6189ms  1.6189ms  cudaLaunchKernel
                    0.00%  25.500us       114     223ns       0ns  3.7000us  cuDeviceGetAttribute
                    0.00%  10.700us         1  10.700us  10.700us  10.700us  cuLibraryUnload
                    0.00%  4.3000us         1  4.3000us  4.3000us  4.3000us  cudaGetDeviceProperties
                    0.00%  3.0000us         1  3.0000us  3.0000us  3.0000us  cuDeviceTotalMem
                    0.00%  2.0000us         3     666ns       0ns  1.7000us  cuDeviceGetCount
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuModuleGetLoadingMode
                    0.00%  1.1000us         1  1.1000us  1.1000us  1.1000us  cuDeviceGetName
                    0.00%  1.0000us         2     500ns       0ns  1.0000us  cuDeviceGet
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
*/
