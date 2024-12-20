#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

/*
 * Various memory access pattern optimizations applied to a matrix transpose
 * kernel.
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

#define BDIMX 16
#define BDIMY 16

void initialData(float* in, const int size) {
    for (int i = 0; i < size; i++) {
        in[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void printData(float* in, const int size) {
    for (int i = 0; i < size; i++) {
        printf("[host] %dth element: %f\n", i, in[i]);
    }
}

void checkResult(float* hostRef, float* gpuRef, const int size, int showme) {
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < size; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("[host] different on %dth element: host %f gpu %f\n", i, hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (!match)  printf("Arrays do not match.\n\n");
}

void transposeHost(float* out, float* in, const int nx, const int ny) {
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            out[ix * ny + iy] = in[iy * nx + ix];
        }
    }
}

__global__ void warmup(float* out, float* in, const int nx, const int ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[iy * nx + ix] = in[iy * nx + ix];
    }
}

// case 0 copy kernel: access data in rows
__global__ void copyRow(float* out, float* in, const int nx, const int ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[iy * nx + ix] = in[iy * nx + ix];
    }
}

// case 1 copy kernel: access data in columns
__global__ void copyCol(float* out, float* in, const int nx, const int ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[ix * ny + iy] = in[ix * ny + iy];
    }
}

// case 2 transpose kernel: read in rows and write in columns
__global__ void transposeNaiveRow(float* out, float* in, const int nx, const int ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

// case 3 transpose kernel: read in columns and write in rows
__global__ void transposeNaiveCol(float* out, float* in, const int nx, const int ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}

// case 4 transpose kernel: read in rows and write in columns + unroll 4 blocks
__global__ void transposeUnroll4Row(float* out, float* in, const int nx, const int ny) {
    unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int ti = iy * nx + ix; // access in rows
    unsigned int to = ix * ny + iy; // access in columns

    if (ix + 3 * blockDim.x < nx && iy < ny) {
        out[to] = in[ti];
        out[to + ny * blockDim.x] = in[ti + blockDim.x];
        out[to + ny * 2 * blockDim.x] = in[ti + 2 * blockDim.x];
        out[to + ny * 3 * blockDim.x] = in[ti + 3 * blockDim.x];
    }
}

// case 5 transpose kernel: read in columns and write in rows + unroll 4 blocks
__global__ void transposeUnroll4Col(float* out, float* in, const int nx, const int ny) {
    unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int ti = iy * nx + ix; // access in rows
    unsigned int to = ix * ny + iy; // access in columns

    if (ix + 3 * blockDim.x < nx && iy < ny) {
        out[ti] = in[to];
        out[ti + blockDim.x] = in[to + blockDim.x * ny];
        out[ti + 2 * blockDim.x] = in[to + 2 * blockDim.x * ny];
        out[ti + 3 * blockDim.x] = in[to + 3 * blockDim.x * ny];
    }
}

/*
 * case 6 :  transpose kernel: read in rows and write in colunms + diagonal coordinate transform
 */
__global__ void transposeDiagonalRow(float* out, float* in, const int nx, const int ny) {
    unsigned int blk_y = blockIdx.x;
    unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    unsigned int ix = blockDim.x * blk_x + threadIdx.x;
    unsigned int iy = blockDim.y * blk_y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

/*
 * case 7 :  transpose kernel: read in columns and write in row + diagonal coordinate transform.
 */
__global__ void transposeDiagonalCol(float* out, float* in, const int nx, const int ny) {
    unsigned int blk_y = blockIdx.x;
    unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    unsigned int ix = blockDim.x * blk_x + threadIdx.x;
    unsigned int iy = blockDim.y * blk_y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}

// main functions
int main(int argc, char** argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("[host] %s starting transpose at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up array size 8092
    int nx = 1 << 13;
    int ny = 1 << 13;

    // select a kernel and block size
    int iKernel = 0;
    int blockx = 16;
    int blocky = 16;

    if (argc > 1) iKernel = atoi(argv[1]);
    if (argc > 2) blockx = atoi(argv[2]);
    if (argc > 3) blocky = atoi(argv[3]);
    if (argc > 4) nx = atoi(argv[4]);
    if (argc > 5) ny = atoi(argv[5]);

    printf(" with matrix nx %d ny %d with blockx %d blocky %d with kernel %d\n", 
        nx, ny, blockx, blocky, iKernel);
    size_t nBytes = nx * ny * sizeof(float);

    // execution configuration
    dim3 block(blockx, blocky);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // allocate host memory
    float* h_A = (float*)malloc(nBytes);
    float* hostRef = (float*)malloc(nBytes);
    float* gpuRef = (float*)malloc(nBytes);

    // initialize host array
    initialData(h_A, nx * ny);

    // transpose at host side
    transposeHost(hostRef, h_A, nx, ny);

    // allocate device memory
    float* d_A, * d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    // warmup to avoide startup overhead
    warmup << <grid, block >> > (d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // kernel pointer and descriptor
    void (*kernel)(float*, float*, int, int);
    char* kernelName;

    // set up kernel
    switch (iKernel)
    {
    case 0:
        kernel = &copyRow;
        kernelName = "CopyRow       ";
        break;

    case 1:
        kernel = &copyCol;
        kernelName = "CopyCol       ";
        break;

    case 2:
        kernel = &transposeNaiveRow;
        kernelName = "NaiveRow      ";
        break;

    case 3:
        kernel = &transposeNaiveCol;
        kernelName = "NaiveCol      ";
        break;

    case 4:
        kernel = &transposeUnroll4Row;
        kernelName = "Unroll4Row    ";
        grid.x = (nx + block.x * 4 - 1) / (block.x * 4);
        break;

    case 5:
        kernel = &transposeUnroll4Col;
        kernelName = "Unroll4Col    ";
        grid.x = (nx + block.x * 4 - 1) / (block.x * 4);
        break;

    case 6:
        kernel = &transposeDiagonalRow;
        kernelName = "DiagonalRow   ";
        break;

    case 7:
        kernel = &transposeDiagonalCol;
        kernelName = "DiagonalCol   ";
        break;
    }

    // run kernel
    kernel << <grid, block >> > (d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // check kernel results
    if (iKernel > 1) {
        CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
        checkResult(hostRef, gpuRef, nx * ny, 1);
    }

    // free host and device memory
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
c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 0
==3320== NVPROF is profiling process 3320, command: ./Cuda.exe 0
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450  with matrix nx 8192 ny 8192 with blockx 16 blocky 16 with kernel 0
==3320== Profiling application: ./Cuda.exe 0
==3320== Warning: 28 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==3320== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   79.43%  87.547ms         1  87.547ms  87.547ms  87.547ms  [CUDA memcpy HtoD]
                   10.29%  11.338ms         1  11.338ms  11.338ms  11.338ms  copyRow(float*, float*, int, int)
                   10.29%  11.338ms         1  11.338ms  11.338ms  11.338ms  warmup(float*, float*, int, int)
      API calls:   43.13%  101.57ms         1  101.57ms  101.57ms  101.57ms  cudaMemcpy
                   29.95%  70.532ms         1  70.532ms  70.532ms  70.532ms  cudaSetDevice
                   13.45%  31.684ms         1  31.684ms  31.684ms  31.684ms  cudaDeviceReset
                    9.63%  22.689ms         2  11.345ms  11.343ms  11.346ms  cudaDeviceSynchronize
                    2.37%  5.5876ms         2  2.7938ms  2.5918ms  2.9958ms  cudaFree
                    0.76%  1.7851ms         2  892.55us  685.30us  1.0998ms  cudaMalloc
                    0.69%  1.6244ms         2  812.20us  32.300us  1.5921ms  cudaLaunchKernel
                    0.01%  22.300us       114     195ns       0ns  4.3000us  cuDeviceGetAttribute
                    0.01%  20.700us         1  20.700us  20.700us  20.700us  cuLibraryUnload
                    0.00%  3.2000us         1  3.2000us  3.2000us  3.2000us  cudaGetDeviceProperties
                    0.00%  2.2000us         3     733ns     100ns  1.8000us  cuDeviceGetCount
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuModuleGetLoadingMode
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuDeviceTotalMem
                    0.00%  1.1000us         2     550ns     100ns  1.0000us  cuDeviceGet
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetName
                    0.00%     700ns         2     350ns     100ns     600ns  cudaGetLastError
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 1
==14696== NVPROF is profiling process 14696, command: ./Cuda.exe 1
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450  with matrix nx 8192 ny 8192 with blockx 16 blocky 16 with kernel 1
==14696== Profiling application: ./Cuda.exe 1
==14696== Warning: 31 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==14696== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   68.15%  88.934ms         1  88.934ms  88.934ms  88.934ms  [CUDA memcpy HtoD]
                   23.16%  30.222ms         1  30.222ms  30.222ms  30.222ms  copyCol(float*, float*, int, int)
                    8.69%  11.338ms         1  11.338ms  11.338ms  11.338ms  warmup(float*, float*, int, int)
      API calls:   40.37%  103.87ms         1  103.87ms  103.87ms  103.87ms  cudaMemcpy
                   28.53%  73.395ms         1  73.395ms  73.395ms  73.395ms  cudaSetDevice
                   16.16%  41.575ms         2  20.787ms  11.338ms  30.237ms  cudaDeviceSynchronize
                   11.81%  30.393ms         1  30.393ms  30.393ms  30.393ms  cudaDeviceReset
                    2.09%  5.3679ms         2  2.6840ms  2.5398ms  2.8281ms  cudaFree
                    0.51%  1.3173ms         2  658.65us  649.80us  667.50us  cudaMalloc
                    0.51%  1.3096ms         2  654.80us  11.600us  1.2980ms  cudaLaunchKernel
                    0.01%  22.200us         1  22.200us  22.200us  22.200us  cuLibraryUnload
                    0.01%  18.600us       114     163ns       0ns  2.8000us  cuDeviceGetAttribute
                    0.00%  2.8000us         1  2.8000us  2.8000us  2.8000us  cudaGetDeviceProperties
                    0.00%  1.9000us         3     633ns     100ns  1.6000us  cuDeviceGetCount
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuDeviceTotalMem
                    0.00%  1.5000us         1  1.5000us  1.5000us  1.5000us  cuModuleGetLoadingMode
                    0.00%  1.0000us         2     500ns     400ns     600ns  cudaGetLastError
                    0.00%     900ns         2     450ns     100ns     800ns  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 2
==6672== NVPROF is profiling process 6672, command: ./Cuda.exe 2
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450  with matrix nx 8192 ny 8192 with blockx 16 blocky 16 with kernel 2
==6672== Profiling application: ./Cuda.exe 2
==6672== Warning: 31 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==6672== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   42.62%  87.058ms         1  87.058ms  87.058ms  87.058ms  [CUDA memcpy HtoD]
                   41.64%  85.049ms         1  85.049ms  85.049ms  85.049ms  [CUDA memcpy DtoH]
                   10.19%  20.819ms         1  20.819ms  20.819ms  20.819ms  transposeNaiveRow(float*, float*, int, int)
                    5.55%  11.337ms         1  11.337ms  11.337ms  11.337ms  warmup(float*, float*, int, int)
      API calls:   57.50%  188.46ms         2  94.231ms  85.423ms  103.04ms  cudaMemcpy
                   20.60%  67.523ms         1  67.523ms  67.523ms  67.523ms  cudaSetDevice
                    9.84%  32.238ms         2  16.119ms  11.412ms  20.826ms  cudaDeviceSynchronize
                    9.78%  32.069ms         1  32.069ms  32.069ms  32.069ms  cudaDeviceReset
                    1.43%  4.6785ms         2  2.3393ms  1.9641ms  2.7144ms  cudaFree
                    0.43%  1.4179ms         2  708.95us  36.500us  1.3814ms  cudaLaunchKernel
                    0.41%  1.3417ms         2  670.85us  669.30us  672.40us  cudaMalloc
                    0.01%  23.600us         1  23.600us  23.600us  23.600us  cuLibraryUnload
                    0.01%  18.800us       114     164ns       0ns  2.6000us  cuDeviceGetAttribute
                    0.00%  4.9000us         1  4.9000us  4.9000us  4.9000us  cudaGetDeviceProperties
                    0.00%  2.3000us         3     766ns     100ns  1.9000us  cuDeviceGetCount
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuDeviceTotalMem
                    0.00%  1.7000us         1  1.7000us  1.7000us  1.7000us  cuModuleGetLoadingMode
                    0.00%  1.0000us         2     500ns       0ns  1.0000us  cuDeviceGet
                    0.00%     700ns         1     700ns     700ns     700ns  cuDeviceGetName
                    0.00%     600ns         2     300ns     100ns     500ns  cudaGetLastError
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 3
==12016== NVPROF is profiling process 12016, command: ./Cuda.exe 3
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450  with matrix nx 8192 ny 8192 with blockx 16 blocky 16 with kernel 3
==12016== Profiling application: ./Cuda.exe 3
==12016== Warning: 35 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==12016== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   42.99%  88.510ms         1  88.510ms  88.510ms  88.510ms  [CUDA memcpy HtoD]
                   41.51%  85.457ms         1  85.457ms  85.457ms  85.457ms  [CUDA memcpy DtoH]
                    9.99%  20.564ms         1  20.564ms  20.564ms  20.564ms  transposeNaiveCol(float*, float*, int, int)
                    5.51%  11.333ms         1  11.333ms  11.333ms  11.333ms  warmup(float*, float*, int, int)
      API calls:   56.22%  186.97ms         2  93.483ms  85.857ms  101.11ms  cudaMemcpy
                   20.14%  66.990ms         1  66.990ms  66.990ms  66.990ms  cudaSetDevice
                    9.96%  33.137ms         1  33.137ms  33.137ms  33.137ms  cudaDeviceReset
                    9.60%  31.927ms         2  15.964ms  11.337ms  20.591ms  cudaDeviceSynchronize
                    1.81%  6.0270ms         2  3.0135ms  2.9041ms  3.1229ms  cudaFree
                    1.75%  5.8171ms         2  2.9086ms  2.8104ms  3.0067ms  cudaMalloc
                    0.49%  1.6292ms         2  814.60us  12.800us  1.6164ms  cudaLaunchKernel
                    0.01%  28.700us         1  28.700us  28.700us  28.700us  cuLibraryUnload
                    0.01%  20.400us       114     178ns       0ns  4.3000us  cuDeviceGetAttribute
                    0.00%  2.7000us         1  2.7000us  2.7000us  2.7000us  cudaGetDeviceProperties
                    0.00%  2.5000us         1  2.5000us  2.5000us  2.5000us  cuDeviceTotalMem
                    0.00%  2.1000us         3     700ns     100ns  1.7000us  cuDeviceGetCount
                    0.00%  2.1000us         1  2.1000us  2.1000us  2.1000us  cuModuleGetLoadingMode
                    0.00%  1.7000us         2     850ns     100ns  1.6000us  cuDeviceGet
                    0.00%  1.4000us         2     700ns     700ns     700ns  cudaGetLastError
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 4
==26260== NVPROF is profiling process 26260, command: ./Cuda.exe 4
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450  with matrix nx 8192 ny 8192 with blockx 16 blocky 16 with kernel 4
==26260== Profiling application: ./Cuda.exe 4
==26260== Warning: 34 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==26260== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   41.69%  87.327ms         1  87.327ms  87.327ms  87.327ms  [CUDA memcpy HtoD]
                   40.01%  83.813ms         1  83.813ms  83.813ms  83.813ms  [CUDA memcpy DtoH]
                   12.88%  26.978ms         1  26.978ms  26.978ms  26.978ms  transposeUnroll4Row(float*, float*, int, int)
                    5.41%  11.340ms         1  11.340ms  11.340ms  11.340ms  warmup(float*, float*, int, int)
      API calls:   54.85%  184.37ms         2  92.184ms  84.279ms  100.09ms  cudaMemcpy
                   20.96%  70.462ms         1  70.462ms  70.462ms  70.462ms  cudaSetDevice
                   11.41%  38.346ms         2  19.173ms  11.353ms  26.993ms  cudaDeviceSynchronize
                    9.88%  33.203ms         1  33.203ms  33.203ms  33.203ms  cudaDeviceReset
                    1.63%  5.4871ms         2  2.7436ms  2.6062ms  2.8809ms  cudaFree
                    0.87%  2.9085ms         2  1.4543ms  837.50us  2.0710ms  cudaMalloc
                    0.38%  1.2794ms         2  639.70us  52.900us  1.2265ms  cudaLaunchKernel
                    0.01%  21.900us         1  21.900us  21.900us  21.900us  cuLibraryUnload
                    0.01%  18.900us       114     165ns       0ns  2.7000us  cuDeviceGetAttribute
                    0.00%  3.5000us         1  3.5000us  3.5000us  3.5000us  cudaGetDeviceProperties
                    0.00%  2.5000us         1  2.5000us  2.5000us  2.5000us  cuDeviceTotalMem
                    0.00%  2.0000us         3     666ns       0ns  1.8000us  cuDeviceGetCount
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuModuleGetLoadingMode
                    0.00%  1.6000us         2     800ns     700ns     900ns  cudaGetLastError
                    0.00%     900ns         2     450ns     100ns     800ns  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 5
==24964== NVPROF is profiling process 24964, command: ./Cuda.exe 5
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450  with matrix nx 8192 ny 8192 with blockx 16 blocky 16 with kernel 5
==24964== Profiling application: ./Cuda.exe 5
==24964== Warning: 30 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==24964== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   44.83%  98.534ms         1  98.534ms  98.534ms  98.534ms  [CUDA memcpy DtoH]
                   40.52%  89.055ms         1  89.055ms  89.055ms  89.055ms  [CUDA memcpy HtoD]
                    9.49%  20.868ms         1  20.868ms  20.868ms  20.868ms  transposeUnroll4Col(float*, float*, int, int)
                    5.16%  11.339ms         1  11.339ms  11.339ms  11.339ms  warmup(float*, float*, int, int)
      API calls:   58.87%  201.13ms         2  100.57ms  99.055ms  102.08ms  cudaMemcpy
                   19.48%  66.556ms         1  66.556ms  66.556ms  66.556ms  cudaSetDevice
                    9.45%  32.284ms         2  16.142ms  11.397ms  20.887ms  cudaDeviceSynchronize
                    9.37%  32.022ms         1  32.022ms  32.022ms  32.022ms  cudaDeviceReset
                    1.68%  5.7530ms         2  2.8765ms  2.6987ms  3.0543ms  cudaFree
                    0.66%  2.2451ms         2  1.1226ms  642.20us  1.6029ms  cudaMalloc
                    0.46%  1.5806ms         2  790.30us  72.100us  1.5085ms  cudaLaunchKernel
                    0.01%  23.900us         1  23.900us  23.900us  23.900us  cuLibraryUnload
                    0.01%  18.500us       114     162ns       0ns  2.6000us  cuDeviceGetAttribute
                    0.00%  3.8000us         1  3.8000us  3.8000us  3.8000us  cudaGetDeviceProperties
                    0.00%  2.4000us         1  2.4000us  2.4000us  2.4000us  cuDeviceTotalMem
                    0.00%  2.3000us         1  2.3000us  2.3000us  2.3000us  cuModuleGetLoadingMode
                    0.00%  2.0000us         3     666ns     100ns  1.6000us  cuDeviceGetCount
                    0.00%  1.2000us         2     600ns     400ns     800ns  cudaGetLastError
                    0.00%  1.0000us         1  1.0000us  1.0000us  1.0000us  cuDeviceGetName
                    0.00%     800ns         2     400ns     100ns     700ns  cuDeviceGet
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 6
==24672== NVPROF is profiling process 24672, command: ./Cuda.exe 6
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450  with matrix nx 8192 ny 8192 with blockx 16 blocky 16 with kernel 6
==24672== Profiling application: ./Cuda.exe 6
==24672== Warning: 31 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==24672== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   40.98%  87.122ms         1  87.122ms  87.122ms  87.122ms  [CUDA memcpy HtoD]
                   39.63%  84.252ms         1  84.252ms  84.252ms  84.252ms  [CUDA memcpy DtoH]
                   14.05%  29.874ms         1  29.874ms  29.874ms  29.874ms  transposeDiagonalRow(float*, float*, int, int)
                    5.33%  11.334ms         1  11.334ms  11.334ms  11.334ms  warmup(float*, float*, int, int)
      API calls:   55.36%  185.54ms         2  92.768ms  84.750ms  100.79ms  cudaMemcpy
                   19.48%  65.292ms         1  65.292ms  65.292ms  65.292ms  cudaSetDevice
                   12.30%  41.234ms         2  20.617ms  11.343ms  29.891ms  cudaDeviceSynchronize
                   10.14%  33.971ms         1  33.971ms  33.971ms  33.971ms  cudaDeviceReset
                    1.74%  5.8308ms         2  2.9154ms  2.7647ms  3.0661ms  cudaFree
                    0.56%  1.8652ms         2  932.60us  573.90us  1.2913ms  cudaMalloc
                    0.40%  1.3523ms         2  676.15us  22.700us  1.3296ms  cudaLaunchKernel
                    0.01%  26.800us       114     235ns       0ns  8.2000us  cuDeviceGetAttribute
                    0.01%  23.700us         1  23.700us  23.700us  23.700us  cuLibraryUnload
                    0.00%  3.7000us         1  3.7000us  3.7000us  3.7000us  cudaGetDeviceProperties
                    0.00%  1.9000us         3     633ns     100ns  1.6000us  cuDeviceGetCount
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuModuleGetLoadingMode
                    0.00%  1.8000us         1  1.8000us  1.8000us  1.8000us  cuDeviceTotalMem
                    0.00%     900ns         2     450ns       0ns     900ns  cuDeviceGet
                    0.00%     900ns         1     900ns     900ns     900ns  cuDeviceGetName
                    0.00%     800ns         2     400ns     300ns     500ns  cudaGetLastError
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid

c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe 7
==16732== NVPROF is profiling process 16732, command: ./Cuda.exe 7
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450  with matrix nx 8192 ny 8192 with blockx 16 blocky 16 with kernel 7
==16732== Profiling application: ./Cuda.exe 7
==16732== Warning: 36 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==16732== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   40.98%  87.035ms         1  87.035ms  87.035ms  87.035ms  [CUDA memcpy HtoD]
                   39.52%  83.949ms         1  83.949ms  83.949ms  83.949ms  [CUDA memcpy DtoH]
                   14.14%  30.035ms         1  30.035ms  30.035ms  30.035ms  transposeDiagonalCol(float*, float*, int, int)
                    5.36%  11.379ms         1  11.379ms  11.379ms  11.379ms  warmup(float*, float*, int, int)
      API calls:   55.26%  186.04ms         2  93.020ms  84.365ms  101.68ms  cudaMemcpy
                   20.13%  67.763ms         1  67.763ms  67.763ms  67.763ms  cudaSetDevice
                   12.31%  41.437ms         2  20.719ms  11.384ms  30.054ms  cudaDeviceSynchronize
                    9.84%  33.118ms         1  33.118ms  33.118ms  33.118ms  cudaDeviceReset
                    1.67%  5.6356ms         2  2.8178ms  2.7644ms  2.8712ms  cudaFree
                    0.40%  1.3402ms         2  670.10us  23.700us  1.3165ms  cudaLaunchKernel
                    0.38%  1.2877ms         2  643.85us  642.30us  645.40us  cudaMalloc
                    0.01%  21.900us         1  21.900us  21.900us  21.900us  cuLibraryUnload
                    0.01%  20.700us       114     181ns       0ns  3.6000us  cuDeviceGetAttribute
                    0.00%  3.6000us         1  3.6000us  3.6000us  3.6000us  cudaGetDeviceProperties
                    0.00%  2.0000us         3     666ns     100ns  1.7000us  cuDeviceGetCount
                    0.00%  2.0000us         1  2.0000us  2.0000us  2.0000us  cuDeviceTotalMem
                    0.00%  1.9000us         1  1.9000us  1.9000us  1.9000us  cuModuleGetLoadingMode
                    0.00%  1.3000us         2     650ns     600ns     700ns  cudaGetLastError
                    0.00%  1.0000us         1  1.0000us  1.0000us  1.0000us  cuDeviceGetName
                    0.00%     800ns         2     400ns       0ns     800ns  cuDeviceGet
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid
*/
