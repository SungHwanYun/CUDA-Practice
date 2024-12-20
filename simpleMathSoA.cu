#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

/*
 * A simple example of using a structore of arrays to store data on the device.
 * This example is used to study the impact on performance of data layout on the
 * GPU.
 *
 * SoA: contiguous reads for x and y
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

struct InnerArray {
    float x[LEN];
    float y[LEN];
};

// functions for inner array outer struct
void initialInnerArray(InnerArray* ip, int size) {
    for (int i = 0; i < size; i++) {
        ip->x[i] = (float)(rand() & 0xFF) / 100.0f;
        ip->y[i] = (float)(rand() & 0xFF) / 100.0f;
    }
}

void testInnerArrayHost(InnerArray* A, InnerArray* C, const int n) {
    for (int idx = 0; idx < n; idx++) {
        C->x[idx] = A->x[idx] + 10.f;
        C->y[idx] = A->y[idx] + 20.f;
    }
}

void printfHostResult(InnerArray* C, const int n) {
    for (int idx = 0; idx < n; idx++) {
        printf("[host] printout idx %d:  x %f y %f\n", idx, C->x[idx], C->y[idx]);
    }
}

void checkInnerArray(InnerArray* hostRef, InnerArray* gpuRef, const int N) {
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++) {
        if (abs(hostRef->x[i] - gpuRef->x[i]) > epsilon) {
            match = 0;
            printf("[host] different on x %dth element: host %f gpu %f\n", i,
                hostRef->x[i], gpuRef->x[i]);
            break;
        }

        if (abs(hostRef->y[i] - gpuRef->y[i]) > epsilon) {
            match = 0;
            printf("[host] different on y %dth element: host %f gpu %f\n", i,
                hostRef->y[i], gpuRef->y[i]);
            break;
        }
    }
    if (!match)  printf("Arrays do not match.\n\n");
}

__global__ void testInnerArray(InnerArray* data, InnerArray* result, const int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float tmpx = data->x[i];
        float tmpy = data->y[i];

        tmpx += 10.f;
        tmpy += 20.f;
        result->x[i] = tmpx;
        result->y[i] = tmpy;
    }
}

__global__ void warmup2(InnerArray* data, InnerArray* result, const int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float tmpx = data->x[i];
        float tmpy = data->y[i];
        tmpx += 10.f;
        tmpy += 20.f;
        result->x[i] = tmpx;
        result->y[i] = tmpy;
    }
}

// test for array of struct
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
    size_t nBytes = sizeof(InnerArray);
    InnerArray* h_A = (InnerArray*)malloc(nBytes);
    InnerArray* hostRef = (InnerArray*)malloc(nBytes);
    InnerArray* gpuRef = (InnerArray*)malloc(nBytes);

    // initialize host array
    initialInnerArray(h_A, nElem);
    testInnerArrayHost(h_A, hostRef, nElem);

    // allocate device memory
    InnerArray* d_A, * d_C;
    CHECK(cudaMalloc((InnerArray**)&d_A, nBytes));
    CHECK(cudaMalloc((InnerArray**)&d_C, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    int blocksize = 128;
    if (argc > 1) blocksize = atoi(argv[1]);

    // execution configuration
    dim3 block(blocksize, 1);
    dim3 grid((nElem + block.x - 1) / block.x, 1);

    // kernel 1:
    warmup2 << <grid, block >> > (d_A, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkInnerArray(hostRef, gpuRef, nElem);
    CHECK(cudaGetLastError());

    testInnerArray << <grid, block >> > (d_A, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkInnerArray(hostRef, gpuRef, nElem);
    CHECK(cudaGetLastError());

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
==16388== NVPROF is profiling process 16388, command: ./Cuda.exe
[host] ./Cuda.exe test struct of array at device 0: NVIDIA GeForce MX450
==16388== Profiling application: ./Cuda.exe
==16388== Warning: 18 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==16388== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   60.65%  21.168ms         2  10.584ms  10.403ms  10.765ms  [CUDA memcpy DtoH]
                   31.38%  10.953ms         1  10.953ms  10.953ms  10.953ms  [CUDA memcpy HtoD]
                    3.99%  1.3920ms         1  1.3920ms  1.3920ms  1.3920ms  testInnerArray(InnerArray*, InnerArray*, int)
                    3.99%  1.3917ms         1  1.3917ms  1.3917ms  1.3917ms  warmup2(InnerArray*, InnerArray*, int)
      API calls:   48.83%  69.392ms         1  69.392ms  69.392ms  69.392ms  cudaSetDevice
                   24.37%  34.635ms         3  11.545ms  10.741ms  12.748ms  cudaMemcpy
                   22.76%  32.352ms         1  32.352ms  32.352ms  32.352ms  cudaDeviceReset
                    2.08%  2.9501ms         2  1.4751ms  1.4018ms  1.5483ms  cudaDeviceSynchronize
                    0.87%  1.2308ms         2  615.40us  77.900us  1.1529ms  cudaLaunchKernel
                    0.78%  1.1049ms         2  552.45us  365.20us  739.70us  cudaFree
                    0.28%  394.10us         2  197.05us  133.40us  260.70us  cudaMalloc
                    0.02%  32.200us       114     282ns       0ns  8.1000us  cuDeviceGetAttribute
                    0.01%  10.300us         1  10.300us  10.300us  10.300us  cuLibraryUnload
                    0.00%  3.4000us         1  3.4000us  3.4000us  3.4000us  cudaGetDeviceProperties
                    0.00%  2.6000us         3     866ns     100ns  2.2000us  cuDeviceGetCount
                    0.00%  2.2000us         2  1.1000us  1.1000us  1.1000us  cudaGetLastError
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuDeviceTotalMem
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuModuleGetLoadingMode
                    0.00%     900ns         2     450ns     100ns     800ns  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
*/
