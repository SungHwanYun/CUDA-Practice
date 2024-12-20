#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

/*
 * A simple example of using an array of structures to store data on the device.
 * This example is used to study the impact on performance of data layout on the
 * GPU.
 *
 * AoS: one contiguous 64-bit read to get x and y (up to 300 cycles)
 */

#define LEN 1<<22
#define CHECK(call) \
{ \
    const cudaError_t error = call;\
    if (error != cudaSuccess) { \
        printf("[device] Error: %s %d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(-10*error); \
    } \
}
struct innerStruct {
    float x;
    float y;
};

struct innerArray {
    float x[LEN];
    float y[LEN];
};

void initialInnerStruct(innerStruct* ip, int size) {
    for (int i = 0; i < size; i++)     {
        ip[i].x = (float)(rand() & 0xFF) / 100.0f;
        ip[i].y = (float)(rand() & 0xFF) / 100.0f;
    }
}

void testInnerStructHost(innerStruct* A, innerStruct* C, const int n) {
    for (int idx = 0; idx < n; idx++) {
        C[idx].x = A[idx].x + 10.f;
        C[idx].y = A[idx].y + 20.f;
    }
}

void checkInnerStruct(innerStruct* hostRef, innerStruct* gpuRef, const int N) {
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i].x - gpuRef[i].x) > epsilon) {
            match = 0;
            printf("[host] different on %dth element: host %f gpu %f\n", 
                i, hostRef[i].x, gpuRef[i].x);
            break;
        }
        if (abs(hostRef[i].y - gpuRef[i].y) > epsilon) {
            match = 0;
            printf("[host] different on %dth element: host %f gpu %f\n", 
                i, hostRef[i].y, gpuRef[i].y);
            break;
        }
    }
    if (!match)  printf("[host] Arrays do not match.\n\n");
}

__global__ void testInnerStruct(innerStruct* data, innerStruct* result, const int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        innerStruct tmp = data[i];
        tmp.x += 10.f;
        tmp.y += 20.f;
        result[i] = tmp;
    }
}

__global__ void warmup(innerStruct* data, innerStruct* result, const int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        innerStruct tmp = data[i];
        tmp.x += 10.f;
        tmp.y += 20.f;
        result[i] = tmp;
    }
}

int main(int argc, char** argv) {
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("[host] %s test struct of array at ", argv[0]);
    printf("device %d: %s \n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // allocate host memory
    int nElem = LEN;
    size_t nBytes = nElem * sizeof(innerStruct);
    innerStruct* h_A = (innerStruct*)malloc(nBytes);
    innerStruct* hostRef = (innerStruct*)malloc(nBytes);
    innerStruct* gpuRef = (innerStruct*)malloc(nBytes);

    // initialize host array
    initialInnerStruct(h_A, nElem);
    testInnerStructHost(h_A, hostRef, nElem);

    // allocate device memory
    innerStruct* d_A, * d_C;
    CHECK(cudaMalloc((innerStruct**)&d_A, nBytes));
    CHECK(cudaMalloc((innerStruct**)&d_C, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    // set up offset for summaryAU: It is blocksize not offset. Thanks.CZ
    int blocksize = 128;

    if (argc > 1) blocksize = atoi(argv[1]);

    // execution configuration
    dim3 block(blocksize, 1);
    dim3 grid((nElem + block.x - 1) / block.x, 1);

    // kernel 1: warmup
    warmup << <grid, block >> > (d_A, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkInnerStruct(hostRef, gpuRef, nElem);
    CHECK(cudaGetLastError());

    // kernel 2: testInnerStruct
    testInnerStruct << <grid, block >> > (d_A, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkInnerStruct(hostRef, gpuRef, nElem);
    CHECK(cudaGetLastError());

    // free memories both host and device
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(hostRef);
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}

/*
output:
c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe
==25996== NVPROF is profiling process 25996, command: ./Cuda.exe
[host] ./Cuda.exe test struct of array at device 0: NVIDIA GeForce MX450
==25996== Profiling application: ./Cuda.exe
==25996== Warning: 43 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==25996== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   60.09%  21.214ms         2  10.607ms  10.543ms  10.672ms  [CUDA memcpy DtoH]
                   30.84%  10.889ms         1  10.889ms  10.889ms  10.889ms  [CUDA memcpy HtoD]
                    4.54%  1.6020ms         1  1.6020ms  1.6020ms  1.6020ms  warmup(innerStruct*, innerStruct*, int)
                    4.54%  1.6016ms         1  1.6016ms  1.6016ms  1.6016ms  testInnerStruct(innerStruct*, innerStruct*, int)
      API calls:   52.89%  85.895ms         1  85.895ms  85.895ms  85.895ms  cudaSetDevice
                   21.57%  35.025ms         3  11.675ms  10.974ms  12.948ms  cudaMemcpy
                   20.07%  32.601ms         1  32.601ms  32.601ms  32.601ms  cudaDeviceReset
                    2.29%  3.7206ms         2  1.8603ms  87.700us  3.6329ms  cudaLaunchKernel
                    2.10%  3.4123ms         2  1.7062ms  1.6864ms  1.7259ms  cudaDeviceSynchronize
                    0.77%  1.2541ms         2  627.05us  508.40us  745.70us  cudaFree
                    0.26%  420.00us         2  210.00us  130.10us  289.90us  cudaMalloc
                    0.02%  31.000us         1  31.000us  31.000us  31.000us  cuLibraryUnload
                    0.02%  24.700us         1  24.700us  24.700us  24.700us  cuDeviceTotalMem
                    0.01%  17.200us       114     150ns       0ns  2.1000us  cuDeviceGetAttribute
                    0.00%  2.7000us         1  2.7000us  2.7000us  2.7000us  cudaGetDeviceProperties
                    0.00%  2.3000us         2  1.1500us  1.1000us  1.2000us  cudaGetLastError
                    0.00%  2.1000us         3     700ns     100ns  1.7000us  cuDeviceGetCount
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuModuleGetLoadingMode
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     700ns         2     350ns     100ns     600ns  cuDeviceGet
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
*/
