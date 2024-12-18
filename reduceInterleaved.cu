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
__global__ void reduceNeighboredLess(int* g_idata, int* g_odata, const int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        // convert tid into local array index
        int index = 2 * stride * tid;
        if (index < blockDim.x) {
            idata[index] += idata[index + stride];
        }

        // synchronize within block
        __syncthreads();
    }
    // write result for this block to global memory
    if (tid == 0) {
        g_odata[blockIdx.x] = idata[0];
    }
}__global__ void reduceNeighbored(int* g_idata, int* g_odata, const int n) {
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
__global__ void reduceInterleaved(int* g_idata, int* g_odata, const int n) {

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
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
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
    printf("[host] gpu reduceNeighbored : grid(%d, %d), block(%d, %d), cpu_sum=%d, gpu_sum=%d\n",
        grid.x, grid.y, block.x, block.y, cpu_sum, gpu_sum);

    // kernel 2: reduceNeighboredLess
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    reduceNeighboredLess << <grid, block >> > (d_idata, d_odata, size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    printf("[host] gpu reduceNeighboredLess : grid(%d, %d), block(%d, %d), cpu_sum=%d, gpu_sum=%d\n",
        grid.x, grid.y, block.x, block.y, cpu_sum, gpu_sum);

    // kernel 3: reduceInterleaved
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    reduceInterleaved << <grid, block >> > (d_idata, d_odata, size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    printf("[host] gpu reduceInterleaved : grid(%d, %d), block(%d, %d), cpu_sum=%d, gpu_sum=%d\n",
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
==24436== NVPROF is profiling process 24436, command: ./Cuda
[host] Using Device 0: NVIDIA GeForce MX450
[host] Data size : 16777216
[host] Execution configure : grid(32768, 1), block(512, 1)
[host] gpu warmup : grid(32768, 1), block(512, 1), cpu_sum=2139095040, gpu_sum=2139095040
[host] gpu reduceNeighbored : grid(32768, 1), block(512, 1), cpu_sum=2139095040, gpu_sum=2139095040
[host] gpu reduceNeighboredLess : grid(32768, 1), block(512, 1), cpu_sum=2139095040, gpu_sum=2139095040
[host] gpu reduceInterleaved : grid(32768, 1), block(512, 1), cpu_sum=2139095040, gpu_sum=2139095040
==24436== Profiling application: ./Cuda
==24436== Warning: 14 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==24436== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   72.81%  90.984ms         4  22.746ms  22.222ms  23.482ms  [CUDA memcpy HtoD]
                   11.07%  13.836ms         1  13.836ms  13.836ms  13.836ms  reduceNeighbored(int*, int*, int)
                    7.80%  9.7509ms         1  9.7509ms  9.7509ms  9.7509ms  reduceNeighboredLess(int*, int*, int)
                    7.49%  9.3621ms         1  9.3621ms  9.3621ms  9.3621ms  warmup(int*, int*, int)
                    0.70%  875.58us         1  875.58us  875.58us  875.58us  reduceInterleaved(int*, int*, int)
                    0.13%  156.96us         4  39.240us  38.528us  40.064us  [CUDA memcpy DtoH]
      API calls:   35.38%  90.755ms         8  11.344ms  160.40us  23.252ms  cudaMemcpy
                   28.42%  72.899ms         1  72.899ms  72.899ms  72.899ms  cudaSetDevice
                   14.15%  36.289ms         8  4.5362ms  520.10us  13.913ms  cudaDeviceSynchronize
                   11.58%  29.707ms         1  29.707ms  29.707ms  29.707ms  cudaDeviceReset
                    9.78%  25.089ms         4  6.2722ms  55.400us  24.858ms  cudaLaunchKernel
                    0.52%  1.3405ms         2  670.25us  464.80us  875.70us  cudaFree
                    0.14%  371.10us         2  185.55us  82.500us  288.60us  cudaMalloc
                    0.01%  22.600us         1  22.600us  22.600us  22.600us  cuLibraryUnload
                    0.01%  21.400us       114     187ns       0ns  2.7000us  cuDeviceGetAttribute
                    0.00%  5.5000us         1  5.5000us  5.5000us  5.5000us  cudaGetDeviceProperties
                    0.00%  4.8000us         1  4.8000us  4.8000us  4.8000us  cuDeviceGetName
                    0.00%  2.5000us         1  2.5000us  2.5000us  2.5000us  cuDeviceTotalMem
                    0.00%  2.3000us         3     766ns       0ns  2.0000us  cuDeviceGetCount
                    0.00%  2.2000us         1  2.2000us  2.2000us  2.2000us  cuModuleGetLoadingMode
                    0.00%     900ns         2     450ns     100ns     800ns  cuDeviceGet
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid
*/
