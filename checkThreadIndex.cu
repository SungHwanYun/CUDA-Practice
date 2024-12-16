#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<time.h>
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

void printMatrix(int* C, const int nx, const int ny) {
    int* ic = C;
    printf("[host] Matrix: (%d x %d)\n", nx, ny);
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            printf("%3d", ic[ix]);
        }
        ic += nx;
        printf("\n");
    }
    printf("\n");
}

__global__ void printThreadIndex(int *A, const int nx, const int ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * nx + ix;
    printf("[device] thread_id (%d, %d) block_id (%d, %d) coordinate (%d, %d) "
        "global index %2d ival %2d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y,
        ix, iy, idx, A[idx]);
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

    // set matrix dimension
    int nx = 8;
    int ny = 6;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(int);

    // malloc host memory
    int* h_A;
    h_A = (int*)malloc(nBytes);

    // initialize host matrix with integer
    initialInt(h_A, nxy);
    printMatrix(h_A, nx, ny);

    // malloc device memory
    int* d_MatA;
    cudaMalloc((void**)&d_MatA, nBytes);

    // transfer data from host to device
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);

    // set up execution configuration
    dim3 block(4, 2);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // invoke the kernel
    printThreadIndex << <grid, block >> > (d_MatA, nx, ny);
    cudaDeviceSynchronize();

    // free host and device memory
    cudaFree(d_MatA);
    free(h_A);

    // reset device
    cudaDeviceReset();
    return 0;

}

/*
output:
C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe
[host] ./Cuda.exe Starting...
==21244== NVPROF is profiling process 21244, command: ./Cuda.exe
[host] Using Device 0: NVIDIA GeForce MX450
[host] Vector size 16777216
[host] Matrix: (8 x 6)
  0  1  2  3  4  5  6  7
  8  9 10 11 12 13 14 15
 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31
 32 33 34 35 36 37 38 39
 40 41 42 43 44 45 46 47

[device] thread_id (0, 0) block_id (1, 1) coordinate (4, 2) global index 20 ival 20
[device] thread_id (1, 0) block_id (1, 1) coordinate (5, 2) global index 21 ival 21
[device] thread_id (2, 0) block_id (1, 1) coordinate (6, 2) global index 22 ival 22
[device] thread_id (3, 0) block_id (1, 1) coordinate (7, 2) global index 23 ival 23
[device] thread_id (0, 1) block_id (1, 1) coordinate (4, 3) global index 28 ival 28
[device] thread_id (1, 1) block_id (1, 1) coordinate (5, 3) global index 29 ival 29
[device] thread_id (2, 1) block_id (1, 1) coordinate (6, 3) global index 30 ival 30
[device] thread_id (3, 1) block_id (1, 1) coordinate (7, 3) global index 31 ival 31
[device] thread_id (0, 0) block_id (1, 2) coordinate (4, 4) global index 36 ival 36
[device] thread_id (1, 0) block_id (1, 2) coordinate (5, 4) global index 37 ival 37
[device] thread_id (2, 0) block_id (1, 2) coordinate (6, 4) global index 38 ival 38
[device] thread_id (3, 0) block_id (1, 2) coordinate (7, 4) global index 39 ival 39
[device] thread_id (0, 1) block_id (1, 2) coordinate (4, 5) global index 44 ival 44
[device] thread_id (1, 1) block_id (1, 2) coordinate (5, 5) global index 45 ival 45
[device] thread_id (2, 1) block_id (1, 2) coordinate (6, 5) global index 46 ival 46
[device] thread_id (3, 1) block_id (1, 2) coordinate (7, 5) global index 47 ival 47
[device] thread_id (0, 0) block_id (1, 0) coordinate (4, 0) global index  4 ival  4
[device] thread_id (1, 0) block_id (1, 0) coordinate (5, 0) global index  5 ival  5
[device] thread_id (2, 0) block_id (1, 0) coordinate (6, 0) global index  6 ival  6
[device] thread_id (3, 0) block_id (1, 0) coordinate (7, 0) global index  7 ival  7
[device] thread_id (0, 1) block_id (1, 0) coordinate (4, 1) global index 12 ival 12
[device] thread_id (1, 1) block_id (1, 0) coordinate (5, 1) global index 13 ival 13
[device] thread_id (2, 1) block_id (1, 0) coordinate (6, 1) global index 14 ival 14
[device] thread_id (3, 1) block_id (1, 0) coordinate (7, 1) global index 15 ival 15
[device] thread_id (0, 0) block_id (0, 2) coordinate (0, 4) global index 32 ival 32
[device] thread_id (1, 0) block_id (0, 2) coordinate (1, 4) global index 33 ival 33
[device] thread_id (2, 0) block_id (0, 2) coordinate (2, 4) global index 34 ival 34
[device] thread_id (3, 0) block_id (0, 2) coordinate (3, 4) global index 35 ival 35
[device] thread_id (0, 1) block_id (0, 2) coordinate (0, 5) global index 40 ival 40
[device] thread_id (1, 1) block_id (0, 2) coordinate (1, 5) global index 41 ival 41
[device] thread_id (2, 1) block_id (0, 2) coordinate (2, 5) global index 42 ival 42
[device] thread_id (3, 1) block_id (0, 2) coordinate (3, 5) global index 43 ival 43
[device] thread_id (0, 0) block_id (0, 0) coordinate (0, 0) global index  0 ival  0
[device] thread_id (1, 0) block_id (0, 0) coordinate (1, 0) global index  1 ival  1
[device] thread_id (2, 0) block_id (0, 0) coordinate (2, 0) global index  2 ival  2
[device] thread_id (3, 0) block_id (0, 0) coordinate (3, 0) global index  3 ival  3
[device] thread_id (0, 1) block_id (0, 0) coordinate (0, 1) global index  8 ival  8
[device] thread_id (1, 1) block_id (0, 0) coordinate (1, 1) global index  9 ival  9
[device] thread_id (2, 1) block_id (0, 0) coordinate (2, 1) global index 10 ival 10
[device] thread_id (3, 1) block_id (0, 0) coordinate (3, 1) global index 11 ival 11
[device] thread_id (0, 0) block_id (0, 1) coordinate (0, 2) global index 16 ival 16
[device] thread_id (1, 0) block_id (0, 1) coordinate (1, 2) global index 17 ival 17
[device] thread_id (2, 0) block_id (0, 1) coordinate (2, 2) global index 18 ival 18
[device] thread_id (3, 0) block_id (0, 1) coordinate (3, 2) global index 19 ival 19
[device] thread_id (0, 1) block_id (0, 1) coordinate (0, 3) global index 24 ival 24
[device] thread_id (1, 1) block_id (0, 1) coordinate (1, 3) global index 25 ival 25
[device] thread_id (2, 1) block_id (0, 1) coordinate (2, 3) global index 26 ival 26
[device] thread_id (3, 1) block_id (0, 1) coordinate (3, 3) global index 27 ival 27
==21244== Profiling application: ./Cuda.exe
==21244== Warning: 34 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==21244== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   97.98%  96.287us         1  96.287us  96.287us  96.287us  printThreadIndex(int*, int, int)
                    2.02%  1.9840us         1  1.9840us  1.9840us  1.9840us  [CUDA memcpy HtoD]
      API calls:   77.26%  119.23ms         1  119.23ms  119.23ms  119.23ms  cudaSetDevice
                   16.54%  25.529ms         1  25.529ms  25.529ms  25.529ms  cudaDeviceReset
                    3.43%  5.2938ms         1  5.2938ms  5.2938ms  5.2938ms  cudaDeviceSynchronize
                    2.35%  3.6279ms         1  3.6279ms  3.6279ms  3.6279ms  cudaLaunchKernel
                    0.16%  249.40us         1  249.40us  249.40us  249.40us  cudaMalloc
                    0.12%  183.60us         1  183.60us  183.60us  183.60us  cudaMemcpy
                    0.10%  151.40us         1  151.40us  151.40us  151.40us  cudaFree
                    0.01%  18.800us       114     164ns       0ns  2.9000us  cuDeviceGetAttribute
                    0.01%  12.300us         1  12.300us  12.300us  12.300us  cuLibraryUnload
                    0.01%  11.700us         1  11.700us  11.700us  11.700us  cudaGetDeviceProperties
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cuModuleGetLoadingMode
                    0.00%  2.3000us         3     766ns     100ns  1.9000us  cuDeviceGetCount
                    0.00%  2.3000us         1  2.3000us  2.3000us  2.3000us  cuDeviceTotalMem
                    0.00%  1.2000us         2     600ns     100ns  1.1000us  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
*/
