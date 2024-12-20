#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

/*
 * This example demonstrates using explicit CUDA memory transfer to implement
 * matrix addition. This code contrasts with sumMatrixGPUManaged.cu, where CUDA
 * managed memory is used to remove all explicit memory transfers and abstract
 * away the concept of physicall separate address spaces.
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

void initialData(float* ip, const int size) {
    for (int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void sumMatrixOnHost(float* A, float* B, float* C, const int nx, const int ny) {
    float* ia = A;
    float* ib = B;
    float* ic = C;

    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx;
        ib += nx;
        ic += nx;
    }
}

void checkResult(float* hostRef, float* gpuRef, const int N) {
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("[host] host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (!match) {
        printf("[host] Arrays do not match.\n\n");
    }
}

__global__ void warmup(float* MatA, float* MatB, float* MatC, int nx, int ny) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny) {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

// grid 2D block 2D
__global__ void sumMatrixGPU(float* MatA, float* MatB, float* MatC, int nx, int ny) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny) {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

int main(int argc, char** argv) {
    printf("[host] %s Starting ", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of matrix
    int nx, ny;
    int ishift = 12;

    if (argc > 1) ishift = atoi(argv[1]);
    nx = ny = 1 << ishift;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("[host] Matrix size: nx %d ny %d\n", nx, ny);

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
    float* d_MatA, * d_MatB, * d_MatC;
    CHECK(cudaMalloc((void**)&d_MatA, nBytes));
    CHECK(cudaMalloc((void**)&d_MatB, nBytes));
    CHECK(cudaMalloc((void**)&d_MatC, nBytes));

    // invoke kernel at host side
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // init device data to 0.0f, then warm-up kernel to obtain accurate timing result
    CHECK(cudaMemset(d_MatA, 0.0f, nBytes));
    CHECK(cudaMemset(d_MatB, 0.0f, nBytes));
    warmup << <grid, block >> > (d_MatA, d_MatB, d_MatC, nx, ny);

    // transfer data from host to device
    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

    sumMatrixGPU << <grid, block >> > (d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

    // check kernel error
    CHECK(cudaGetLastError());

    // check device results
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    CHECK(cudaFree(d_MatA));
    CHECK(cudaFree(d_MatB));
    CHECK(cudaFree(d_MatC));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());

    return (0);
}
/*
output:
c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe
[host] ./Cuda.exe Starting ==3796== NVPROF is profiling process 3796, command: ./Cuda.exe
using Device 0: NVIDIA GeForce MX450
[host] Matrix size: nx 4096 ny 4096
==3796== Profiling application: ./Cuda.exe
==3796== Warning: 24 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==3796== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   57.52%  45.704ms         2  22.852ms  22.051ms  23.653ms  [CUDA memcpy HtoD]
                   28.58%  22.707ms         1  22.707ms  22.707ms  22.707ms  [CUDA memcpy DtoH]
                    5.30%  4.2136ms         1  4.2136ms  4.2136ms  4.2136ms  warmup(float*, float*, float*, int, int)
                    5.30%  4.2111ms         1  4.2111ms  4.2111ms  4.2111ms  sumMatrixGPU(float*, float*, float*, int, int)
                    3.30%  2.6220ms         2  1.3110ms  1.3094ms  1.3126ms  [CUDA memset]
      API calls:   36.01%  75.206ms         1  75.206ms  75.206ms  75.206ms  cudaSetDevice
                   34.80%  72.672ms         3  24.224ms  23.069ms  25.862ms  cudaMemcpy
                   11.88%  24.818ms         1  24.818ms  24.818ms  24.818ms  cudaDeviceReset
                   11.76%  24.548ms         2  12.274ms  47.800us  24.500ms  cudaLaunchKernel
                    2.34%  4.8872ms         1  4.8872ms  4.8872ms  4.8872ms  cudaDeviceSynchronize
                    1.41%  2.9464ms         2  1.4732ms  11.800us  2.9346ms  cudaMemset
                    0.92%  1.9204ms         3  640.13us  343.60us  1.2204ms  cudaFree
                    0.83%  1.7299ms         3  576.63us  255.80us  1.0642ms  cudaMalloc
                    0.02%  46.300us       114     406ns       0ns  25.600us  cuDeviceGetAttribute
                    0.02%  36.200us         1  36.200us  36.200us  36.200us  cuLibraryUnload
                    0.00%  4.6000us         1  4.6000us  4.6000us  4.6000us  cudaGetDeviceProperties
                    0.00%  2.1000us         3     700ns     100ns  1.7000us  cuDeviceGetCount
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuModuleGetLoadingMode
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuDeviceTotalMem
                    0.00%  1.0000us         1  1.0000us  1.0000us  1.0000us  cudaGetLastError
                    0.00%     900ns         2     450ns       0ns     900ns  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     600ns         1     600ns     600ns     600ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
*/
