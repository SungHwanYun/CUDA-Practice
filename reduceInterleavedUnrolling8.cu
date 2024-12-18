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
__global__ void reduceInterleavedUnrolling2(int* g_idata, int* g_odata, const int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x * 2;

    // boundary check
    if (idx + blockDim.x >= n) return;
    g_idata[idx] += g_idata[idx + blockDim.x];
    __syncthreads();

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
__global__ void reduceInterleavedUnrolling4(int* g_idata, int* g_odata, const int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x * 4;

    // boundary check
    if (idx + blockDim.x * 3 >= n) return;
    int a1 = g_idata[idx];
    int a2 = g_idata[idx + blockDim.x];
    int a3 = g_idata[idx + blockDim.x * 2];
    int a4 = g_idata[idx + blockDim.x * 3];
    g_idata[idx] = a1 + a2 + a3 + a4;
    __syncthreads();

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
__global__ void reduceInterleavedUnrolling8(int* g_idata, int* g_odata, const int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int* idata = g_idata + blockIdx.x * blockDim.x * 8;

    // boundary check
    if (idx + blockDim.x * 7 >= n) return;
    int a1 = g_idata[idx];
    int a2 = g_idata[idx + blockDim.x];
    int a3 = g_idata[idx + blockDim.x * 2];
    int a4 = g_idata[idx + blockDim.x * 3];
    int a5 = g_idata[idx + blockDim.x * 4];
    int a6 = g_idata[idx + blockDim.x * 5];
    int a7 = g_idata[idx + blockDim.x * 6];
    int a8 = g_idata[idx + blockDim.x * 7];
    g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    __syncthreads();

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
    reduceInterleaved << <grid, block>> > (d_idata, d_odata, size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];
    printf("[host] gpu reduceInterleaved : grid(%d, %d), block(%d, %d), cpu_sum=%d, gpu_sum=%d\n",
        grid.x, grid.y, block.x, block.y, cpu_sum, gpu_sum);

    // kernel 4: reduceInterleavedUnrolling2
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    reduceInterleavedUnrolling2 <<< grid.x / 2, block >>> (d_idata, d_odata, size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 2; i++) gpu_sum += h_odata[i];
    printf("[host] gpu reduceInterleavedUnrolling2 : grid(%d, %d), block(%d, %d), cpu_sum=%d, gpu_sum=%d\n",
        grid.x / 2, grid.y, block.x, block.y, cpu_sum, gpu_sum);

    // kernel 5: reduceInterleavedUnrolling4
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    reduceInterleavedUnrolling4 << < grid.x / 4, block >> > (d_idata, d_odata, size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 4; i++) gpu_sum += h_odata[i];
    printf("[host] gpu reduceInterleavedUnrolling4 : grid(%d, %d), block(%d, %d), cpu_sum=%d, gpu_sum=%d\n",
        grid.x / 4, grid.y, block.x, block.y, cpu_sum, gpu_sum);

    // kernel 6: reduceInterleavedUnrolling8
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    reduceInterleavedUnrolling8 << < grid.x / 8, block >> > (d_idata, d_odata, size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];
    printf("[host] gpu reduceInterleavedUnrolling8 : grid(%d, %d), block(%d, %d), cpu_sum=%d, gpu_sum=%d\n",
        grid.x / 8, grid.y, block.x, block.y, cpu_sum, gpu_sum);

    // free host memory
    free(h_idata); free(h_odata); free(tmp);

    // free device memory
    cudaFree(d_idata); cudaFree(d_odata);

    // reset device
    cudaDeviceReset();
}

/*
output:
C:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe
[host] ./Cuda.exe Starting...
==13632== NVPROF is profiling process 13632, command: ./Cuda.exe
[host] Using Device 0: NVIDIA GeForce MX450
[host] Data size : 16777216
[host] Execution configure : grid(32768, 1), block(512, 1)
[host] gpu warmup : grid(32768, 1), block(512, 1), cpu_sum=2139095040, gpu_sum=2139095040
[host] gpu reduceNeighbored : grid(32768, 1), block(512, 1), cpu_sum=2139095040, gpu_sum=2139095040
[host] gpu reduceNeighboredLess : grid(32768, 1), block(512, 1), cpu_sum=2139095040, gpu_sum=2139095040
[host] gpu reduceInterleaved : grid(32768, 1), block(512, 1), cpu_sum=2139095040, gpu_sum=2139095040
[host] gpu reduceInterleavedUnrolling2 : grid(16384, 1), block(512, 1), cpu_sum=2139095040, gpu_sum=2139095040
[host] gpu reduceInterleavedUnrolling4 : grid(8192, 1), block(512, 1), cpu_sum=2139095040, gpu_sum=2139095040
[host] gpu reduceInterleavedUnrolling8 : grid(4096, 1), block(512, 1), cpu_sum=2139095040, gpu_sum=2139095040
==13632== Profiling application: ./Cuda.exe
==13632== Warning: 7 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==13632== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   73.99%  162.10ms         7  23.158ms  21.766ms  23.997ms  [CUDA memcpy HtoD]
                    6.32%  13.848ms         1  13.848ms  13.848ms  13.848ms  warmup(int*, int*, int)
                    6.32%  13.844ms         1  13.844ms  13.844ms  13.844ms  reduceNeighbored(int*, int*, int)
                    4.46%  9.7615ms         1  9.7615ms  9.7615ms  9.7615ms  reduceNeighboredLess(int*, int*, int)
                    4.27%  9.3643ms         1  9.3643ms  9.3643ms  9.3643ms  reduceInterleaved(int*, int*, int)
                    2.30%  5.0424ms         1  5.0424ms  5.0424ms  5.0424ms  reduceInterleavedUnrolling2(int*, int*, int)
                    1.35%  2.9493ms         1  2.9493ms  2.9493ms  2.9493ms  reduceInterleavedUnrolling4(int*, int*, int)
                    0.86%  1.8914ms         1  1.8914ms  1.8914ms  1.8914ms  reduceInterleavedUnrolling8(int*, int*, int)
                    0.13%  276.70us         7  39.528us  38.592us  41.087us  [CUDA memcpy DtoH]
      API calls:   42.49%  161.19ms        14  11.514ms  115.80us  23.749ms  cudaMemcpy
                   19.42%  73.661ms         1  73.661ms  73.661ms  73.661ms  cudaSetDevice
                   16.10%  61.089ms        14  4.3635ms  515.00us  13.900ms  cudaDeviceSynchronize
                   13.17%  49.956ms         7  7.1365ms  51.500us  49.570ms  cudaLaunchKernel
                    8.35%  31.690ms         1  31.690ms  31.690ms  31.690ms  cudaDeviceReset
                    0.34%  1.2840ms         2  642.00us  505.40us  778.60us  cudaFree
                    0.11%  410.00us         2  205.00us  72.800us  337.20us  cudaMalloc
                    0.01%  40.100us         1  40.100us  40.100us  40.100us  cuLibraryUnload
                    0.01%  26.100us       114     228ns       0ns  3.9000us  cuDeviceGetAttribute
                    0.00%  4.9000us         1  4.9000us  4.9000us  4.9000us  cudaGetDeviceProperties
                    0.00%  2.6000us         1  2.6000us  2.6000us  2.6000us  cuDeviceTotalMem
                    0.00%  2.1000us         3     700ns     100ns  1.6000us  cuDeviceGetCount
                    0.00%  1.6000us         1  1.6000us  1.6000us  1.6000us  cuModuleGetLoadingMode
                    0.00%  1.0000us         2     500ns     100ns     900ns  cuDeviceGet
                    0.00%  1.0000us         1  1.0000us  1.0000us  1.0000us  cuDeviceGetName
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceGetLuid
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetUuid
*/
