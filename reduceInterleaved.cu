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
}
__global__ void reduceInterleaved(int* g_idata, int* g_odata, const int n) {
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
==23752== NVPROF is profiling process 23752, command: ./Cuda
[host] Using Device 0: NVIDIA GeForce MX450
[host] Data size : 16777216
[host] Execution configure : grid(32768, 1), block(512, 1)
[host] gpu warmup : grid(32768, 1), block(512, 1), cpu_sum=2139095040, gpu_sum=2139095040
[host] gpu reduceNeighbored : grid(32768, 1), block(512, 1), cpu_sum=2139095040, gpu_sum=2139095040
[host] gpu reduceNeighboredLess : grid(32768, 1), block(512, 1), cpu_sum=2139095040, gpu_sum=2139095040
[host] gpu reduceInterleaved : grid(32768, 1), block(512, 1), cpu_sum=2139095040, gpu_sum=2139095040
==23752== Profiling application: ./Cuda
==23752== Warning: 31 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==23752== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   66.40%  92.763ms         4  23.191ms  22.453ms  23.942ms  [CUDA memcpy HtoD]
                    9.91%  13.842ms         1  13.842ms  13.842ms  13.842ms  warmup(int*, int*, int)
                    9.90%  13.837ms         1  13.837ms  13.837ms  13.837ms  reduceNeighbored(int*, int*, int)
                    6.98%  9.7502ms         1  9.7502ms  9.7502ms  9.7502ms  reduceNeighboredLess(int*, int*, int)
                    6.70%  9.3624ms         1  9.3624ms  9.3624ms  9.3624ms  reduceInterleaved(int*, int*, int)
                    0.11%  158.46us         4  39.615us  39.232us  40.224us  [CUDA memcpy DtoH]
      API calls:   35.53%  92.507ms         8  11.563ms  209.00us  23.592ms  cudaMemcpy
                   23.64%  61.558ms         1  61.558ms  61.558ms  61.558ms  cudaSetDevice
                   18.90%  49.201ms         8  6.1501ms  513.60us  13.922ms  cudaDeviceSynchronize
                   11.59%  30.185ms         1  30.185ms  30.185ms  30.185ms  cudaDeviceReset
                    9.74%  25.352ms         4  6.3381ms  55.300us  25.174ms  cudaLaunchKernel
                    0.41%  1.0732ms         2  536.60us  352.30us  720.90us  cudaFree
                    0.15%  402.70us         2  201.35us  103.60us  299.10us  cudaMalloc
                    0.01%  27.000us         1  27.000us  27.000us  27.000us  cuLibraryUnload
                    0.01%  18.900us       114     165ns       0ns  2.7000us  cuDeviceGetAttribute
                    0.00%  7.2000us         1  7.2000us  7.2000us  7.2000us  cudaGetDeviceProperties
                    0.00%  3.1000us         3  1.0330us     200ns  2.6000us  cuDeviceGetCount
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cuDeviceTotalMem
                    0.00%  2.3000us         1  2.3000us  2.3000us  2.3000us  cuModuleGetLoadingMode
                    0.00%  1.2000us         2     600ns     200ns  1.0000us  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid
*/
