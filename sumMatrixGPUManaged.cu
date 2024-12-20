#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

/*
 * This example demonstrates the use of CUDA managed memory to implement matrix
 * addition. In this example, arbitrary pointers can be dereferenced on the host
 * and device. CUDA will automatically manage the transfer of data to and from
 * the GPU as needed by the application. There is no need for the programmer to
 * use cudaMemcpy, cudaHostGetDevicePointer, or any other CUDA API involved with
 * explicitly transferring data. In addition, because CUDA managed memory is not
 * forced to reside in a single place it can be transferred to the optimal
 * memory space and not require round-trips over the PCIe bus every time a
 * cross-device reference is performed (as is required with zero copy and UVA).
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
    float* A, * B, * hostRef, * gpuRef;
    CHECK(cudaMallocManaged((void**)&A, nBytes));
    CHECK(cudaMallocManaged((void**)&B, nBytes));
    CHECK(cudaMallocManaged((void**)&gpuRef, nBytes); );
    CHECK(cudaMallocManaged((void**)&hostRef, nBytes););

    // initialize data at host side
    initialData(A, nxy);
    initialData(B, nxy);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    sumMatrixOnHost(A, B, hostRef, nx, ny);

    // invoke kernel at host side
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // warm-up kernel, with unified memory all pages will migrate from host to
    // device
    warmup << <grid, block >> > (A, B, gpuRef, 1, 1);
    sumMatrixGPU << <grid, block >> > (A, B, gpuRef, nx, ny);
    CHECK(cudaDeviceSynchronize());

    // check kernel error
    CHECK(cudaGetLastError());

    // check device results
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    CHECK(cudaFree(A));
    CHECK(cudaFree(B));
    CHECK(cudaFree(hostRef));
    CHECK(cudaFree(gpuRef));

    // reset device
    CHECK(cudaDeviceReset());
}

/*
output:
c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe
[host] ./Cuda.exe Starting ==3268== NVPROF is profiling process 3268, command: ./Cuda.exe
using Device 0: NVIDIA GeForce MX450
[host] Matrix size: nx 4096 ny 4096
==3268== Profiling application: ./Cuda.exe
==3268== Warning: 34 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==3268== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   79.03%  4.2530ms         1  4.2530ms  4.2530ms  4.2530ms  sumMatrixGPU(float*, float*, float*, int, int)
                   20.97%  1.1286ms         1  1.1286ms  1.1286ms  1.1286ms  warmup(float*, float*, float*, int, int)
      API calls:   60.85%  363.26ms         2  181.63ms  31.100us  363.23ms  cudaLaunchKernel
                   15.70%  93.709ms         4  23.427ms  6.1693ms  63.253ms  cudaMallocManaged
                   10.84%  64.684ms         1  64.684ms  64.684ms  64.684ms  cudaSetDevice
                    7.90%  47.182ms         4  11.795ms  4.8670ms  18.414ms  cudaFree
                    3.80%  22.656ms         1  22.656ms  22.656ms  22.656ms  cudaDeviceReset
                    0.91%  5.4125ms         1  5.4125ms  5.4125ms  5.4125ms  cudaDeviceSynchronize
                    0.00%  29.500us         1  29.500us  29.500us  29.500us  cuLibraryUnload
                    0.00%  20.300us       114     178ns       0ns  3.4000us  cuDeviceGetAttribute
                    0.00%  2.8000us         1  2.8000us  2.8000us  2.8000us  cudaGetDeviceProperties
                    0.00%  2.2000us         3     733ns     100ns  1.9000us  cuDeviceGetCount
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuDeviceTotalMem
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuModuleGetLoadingMode
                    0.00%     800ns         2     400ns     100ns     700ns  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     700ns         1     700ns     700ns     700ns  cudaGetLastError
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

==3268== Unified Memory profiling result:
Device "NVIDIA GeForce MX450 (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
    2048  128.00KB  128.00KB  128.00KB  256.0000MB  296.3975ms  Host To Device
    2322  169.34KB  64.000KB  1.0000MB  384.0000MB   2.880703s  Device To Host
*/
