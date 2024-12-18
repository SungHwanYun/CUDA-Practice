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
#include<time.h>
void initialData(int* ip, const int N) {
    // generate different seed for random number
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < N; i++) {
        ip[i] = (int)(rand() & 0xff);
    }
}

int recursiveReduce(int* data, const int size) {
    // terminate check
    if (size == 1) return data[0];

    // renew the stride
    const int stride = size / 2;

    // in-place reduction
    for (int i = 0; i < stride; i++) {
        data[i] += data[i + stride];
    }
    return recursiveReduce(data, stride);
}
__global__ void reduceNeighbored(int* g_idata, int* g_odata, const int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within block
        __syncthreads();
    }
    // write result for this block to global memory
    if (tid == 0) {
        g_odata[blockIdx.x] = idata[0];
    }
}
__global__ void warmup(int* g_idata, int* g_odata, const int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within block
        __syncthreads();
    }
    // write result for this block to global memory
    if (tid == 0) {
        g_odata[blockIdx.x] = idata[0];
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

    // set up data size
    int size = 1 << 24; 
    size_t nBytes = size * sizeof(int);
    printf("[host] Data size : %d\n", size);

    // set up execution configuration
    int blocksize = 512;
    if (argc > 1) {
        blocksize = atoi(argv[1]);
    }
    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("[host] Execution configure : grid(%d, %d), block(%d, %d)\n", 
        grid.x, grid.y, block.x, block.y);

    // allocate host memory
    int* h_idata = (int*)malloc(nBytes);
    int* h_odata = (int*)malloc(grid.x*sizeof(int));
    int* tmp = (int*)malloc(nBytes);

    // initialize the array
    initialData(h_idata, size);
    memcpy(tmp, h_idata, nBytes);

    // allocate device memory
    int* d_idata, * d_odata;
    cudaMalloc((void**)&d_idata, nBytes);
    cudaMalloc((void**)&d_odata, grid.x*sizeof(int));

    // cpu reduction
    int cpu_sum = recursiveReduce(tmp, size);

    // warmup kernel
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    warmup << <grid, block >> > (d_idata, d_odata, size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    int gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    printf("[host] gpu warmup : grid(%d, %d), block(%d, %d), cpu_sum=%d, gpu_sum=%d\n",
        grid.x, grid.y, block.x, block.y, cpu_sum, gpu_sum);

    // kernel 1: reduceNeighbored
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    reduceNeighbored << <grid, block >> > (d_idata, d_odata, size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    printf("[host] gpu Neighbored : grid(%d, %d), block(%d, %d), cpu_sum=%d, gpu_sum=%d\n",
        grid.x, grid.y, block.x, block.y, cpu_sum, gpu_sum);

    // free host memory
    free(h_idata); free(h_odata); free(tmp);

    // free device memory
    cudaFree(d_idata); cudaFree(d_odata);

    // reset device
    cudaDeviceReset();
}

/*
output:
C:\coding\Cuda\x64\Debug>nvprof ./Cuda
[host] ./Cuda Starting...
==26836== NVPROF is profiling process 26836, command: ./Cuda
[host] Using Device 0: NVIDIA GeForce MX450
[host] Data size : 16777216
[host] Execution configure : grid(32768, 1), block(512, 1)
[host] gpu warmup : grid(32768, 1), block(512, 1), cpu_sum=2139095040, gpu_sum=2139095040
[host] gpu Neighbored : grid(32768, 1), block(512, 1), cpu_sum=2139095040, gpu_sum=2139095040
==26836== Profiling application: ./Cuda
==26836== Warning: 30 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==26836== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   61.66%  44.654ms         2  22.327ms  22.097ms  22.558ms  [CUDA memcpy HtoD]
                   19.12%  13.843ms         1  13.843ms  13.843ms  13.843ms  reduceNeighbored(int*, int*, int)
                   19.11%  13.840ms         1  13.840ms  13.840ms  13.840ms  warmup(int*, int*, int)
                    0.11%  79.552us         2  39.776us  39.680us  39.872us  [CUDA memcpy DtoH]
      API calls:   45.20%  87.309ms         1  87.309ms  87.309ms  87.309ms  cudaSetDevice
                   23.18%  44.782ms         4  11.196ms  144.60us  22.255ms  cudaMemcpy
                   15.01%  28.996ms         1  28.996ms  28.996ms  28.996ms  cudaDeviceReset
                   14.94%  28.855ms         4  7.2137ms  498.80us  13.984ms  cudaDeviceSynchronize
                    0.93%  1.8058ms         2  902.90us  53.900us  1.7519ms  cudaLaunchKernel
                    0.53%  1.0164ms         2  508.20us  387.50us  628.90us  cudaFree
                    0.19%  364.60us         2  182.30us  81.000us  283.60us  cudaMalloc
                    0.01%  18.700us       114     164ns       0ns  2.5000us  cuDeviceGetAttribute
                    0.01%  14.400us         1  14.400us  14.400us  14.400us  cuLibraryUnload
                    0.00%  5.1000us         1  5.1000us  5.1000us  5.1000us  cudaGetDeviceProperties
                    0.00%  2.2000us         3     733ns     100ns  1.8000us  cuDeviceGetCount
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuDeviceTotalMem
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuModuleGetLoadingMode
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuDeviceGetName
                    0.00%     900ns         2     450ns     100ns     800ns  cuDeviceGet
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetUuid
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
*/
