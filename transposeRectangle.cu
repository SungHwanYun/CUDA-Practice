#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

/*
 * Example kernels for transposing a rectangular host array using a variety of
 * optimizations, including shared memory, unrolling, and memory padding.
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

  // Some kernels assume square blocks
#define BDIMX 16
#define BDIMY BDIMX

#define INDEX(ROW, COL, INNER) ((ROW) * (INNER) + (COL))
#define IPAD 2

void initialData(float* in, const int size) {
    for (int i = 0; i < size; i++) {
        in[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void printData(float* in, const int size) {
    for (int i = 0; i < size; i++) {
        printf("%3.0f ", in[i]);
    }
    printf("\n");
}

void checkResult(float* hostRef, float* gpuRef, int rows, int cols) {
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int index = INDEX(i, j, cols);
            if (abs(hostRef[index] - gpuRef[index]) > epsilon) {
                match = 0;
                printf("[host] different on (%d, %d) (offset=%d) element in "
                    "transposed matrix: host %f gpu %f\n", i, j, index,
                    hostRef[index], gpuRef[index]);
                break;
            }
        }
        if (!match) break;
    }

    if (!match)  printf("[host] Arrays do not match.\n\n");
}

void transposeHost(float* out, float* in, const int nrows, const int ncols) {
    for (int iy = 0; iy < nrows; ++iy) {
        for (int ix = 0; ix < ncols; ++ix) {
            out[INDEX(ix, iy, nrows)] = in[INDEX(iy, ix, ncols)];
        }
    }
}

__global__ void copyGmem(float* out, float* in, const int nrows, const int ncols) {
    // matrix coordinate (ix,iy)
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    // transpose with boundary test
    if (row < nrows && col < ncols) {
        // NOTE this is a transpose, not a copy
        out[INDEX(col, row, nrows)] = in[INDEX(row, col, ncols)];
    }
}

__global__ void naiveGmem(float* out, float* in, const int nrows, const int ncols) {
    // matrix coordinate (ix,iy)
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    // transpose with boundary test
    if (row < nrows && col < ncols) {
        out[INDEX(col, row, nrows)] = in[INDEX(row, col, ncols)];
    }
}

__global__ void naiveGmemUnroll(float* out, float* in, const int nrows, const int ncols) {
    // Pretend there are twice as many blocks in the x direction
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = (2 * blockIdx.x * blockDim.x) + threadIdx.x;

    if (row < nrows) {
        if (col < ncols) {
            out[INDEX(col, row, nrows)] = in[INDEX(row, col, ncols)];
        }
        col += blockDim.x;
        if (col < ncols) {
            out[INDEX(col, row, nrows)] = in[INDEX(row, col, ncols)];
        }
    }
}

__global__ void transposeSmem(float* out, float* in, int nrows, int ncols) {
    // static shared memory
    __shared__ float tile[BDIMY][BDIMX];

    // coordinate in original matrix
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

    // linear global memory index for original matrix
    unsigned int offset = INDEX(row, col, ncols);

    if (row < nrows && col < ncols) {
        // load data from global memory to shared memory
        tile[threadIdx.y][threadIdx.x] = in[offset];
    }

    // thread index in transposed block
    unsigned int bidx, irow, icol;
    bidx = threadIdx.y * blockDim.x + threadIdx.x;
    irow = bidx / blockDim.y;
    icol = bidx % blockDim.y;

    // NOTE - need to transpose row and col on block and thread-block level:
    // 1. swap blocks x-y
    // 2. swap thread x-y assignment (irow and icol calculations above)
    // note col still has continuous threadIdx.x -> coalesced gst
    col = blockIdx.y * blockDim.y + icol;
    row = blockIdx.x * blockDim.x + irow;

    // linear global memory index for transposed matrix
      // NOTE nrows is stride of result, row and col are transposed
    unsigned int transposed_offset = INDEX(row, col, nrows);
    // thread synchronization
    __syncthreads();

    // NOTE invert sizes for write check
    if (row < ncols && col < nrows) {
        // store data to global memory from shared memory
        out[transposed_offset] = tile[icol][irow]; // NOTE icol,irow not irow,icol
    }
}

__global__ void transposeSmemUnroll(float* out, float* in, const int nrows, const int ncols)
{
    // static 1D shared memory
    __shared__ float tile[BDIMY][BDIMX * 2];

    // coordinate in original matrix
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = (2 * blockIdx.x * blockDim.x) + threadIdx.x;

    unsigned int row2 = row;
    unsigned int col2 = col + blockDim.x;

    // linear global memory index for original matrix
    unsigned int offset = INDEX(row, col, ncols);
    unsigned int offset2 = INDEX(row2, col2, ncols);

    // thread index in transposed block
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    // linear global memory index for transposed matrix
    unsigned int transposed_offset = INDEX(col, row, nrows);
    unsigned int transposed_offset2 = INDEX(col2, row2, nrows);

    if (row < nrows && col < ncols) {
        tile[threadIdx.y][threadIdx.x] = in[offset];
    }
    if (row2 < nrows && col2 < ncols) {
        tile[threadIdx.y][blockDim.x + threadIdx.x] = in[offset2];
    }

    __syncthreads();

    if (row < nrows && col < ncols) {
        out[transposed_offset] = tile[irow][icol];
    }
    if (row2 < nrows && col2 < ncols) {
        out[transposed_offset2] = tile[irow][blockDim.x + icol];
    }
}

__global__ void transposeSmemUnrollPad(float* out, float* in, const int nrows, const int ncols) {
    // static 1D shared memory with padding
    __shared__ float tile[BDIMY][BDIMX * 2 + IPAD];

    // coordinate in original matrix
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = (2 * blockIdx.x * blockDim.x) + threadIdx.x;

    unsigned int row2 = row;
    unsigned int col2 = col + blockDim.x;

    // linear global memory index for original matrix
    unsigned int offset = INDEX(row, col, ncols);
    unsigned int offset2 = INDEX(row2, col2, ncols);

    // thread index in transposed block
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    // linear global memory index for transposed matrix
    unsigned int transposed_offset = INDEX(col, row, nrows);
    unsigned int transposed_offset2 = INDEX(col2, row2, nrows);

    if (row < nrows && col < ncols) {
        tile[threadIdx.y][threadIdx.x] = in[offset];
    }
    if (row2 < nrows && col2 < ncols) {
        tile[threadIdx.y][blockDim.x + threadIdx.x] = in[offset2];
    }

    __syncthreads();

    if (row < nrows && col < ncols) {
        out[transposed_offset] = tile[irow][icol];
    }
    if (row2 < nrows && col2 < ncols) {
        out[transposed_offset2] = tile[irow][blockDim.x + icol];
    }
}

__global__ void transposeSmemUnrollPadDyn(float* out, float* in, const int nrows, const int ncols) {
    // dynamic shared memory
    extern __shared__ float tile[];

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = (2 * blockIdx.x * blockDim.x) + threadIdx.x;

    unsigned int row2 = row;
    unsigned int col2 = col + blockDim.x;

    // linear global memory index for original matrix
    unsigned int offset = INDEX(row, col, ncols);
    unsigned int offset2 = INDEX(row2, col2, ncols);

    // thread index in transposed block
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    // coordinate in transposed matrix
    unsigned int transposed_offset = INDEX(col, row, nrows);
    unsigned int transposed_offset2 = INDEX(col2, row2, nrows);

    if (row < nrows && col < ncols) {
        tile[INDEX(threadIdx.y, threadIdx.x, BDIMX * 2 + IPAD)] = in[offset];
    }
    if (row2 < nrows && col2 < ncols) {
        tile[INDEX(threadIdx.y, blockDim.x + threadIdx.x, BDIMX * 2 + IPAD)] =
            in[offset2];
    }

    __syncthreads();

    if (row < nrows && col < ncols) {
        out[transposed_offset] = tile[INDEX(irow, icol, BDIMX * 2 + IPAD)];
    }
    if (row2 < nrows && col2 < ncols) {
        out[transposed_offset2] = tile[INDEX(irow, blockDim.x + icol, BDIMX * 2 + IPAD)];
    }
}

__global__ void transposeSmemPad(float* out, float* in, int nrows, int ncols) {
    // static shared memory with padding
    __shared__ float tile[BDIMY][BDIMX + IPAD];

    // coordinate in original matrix
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

    // linear global memory index for original matrix
    unsigned int offset = INDEX(row, col, ncols);

    // thread index in transposed block
    unsigned int bidx, irow, icol;
    bidx = threadIdx.y * blockDim.x + threadIdx.x;
    irow = bidx / blockDim.y;
    icol = bidx % blockDim.y;

    // linear global memory index for transposed matrix
    unsigned int transposed_offset = INDEX(col, row, nrows);

    // transpose with boundary test
    if (row < nrows && col < ncols) {
        // load data from global memory to shared memory
        tile[threadIdx.y][threadIdx.x] = in[offset];

        // thread synchronization
        __syncthreads();

        // store data to global memory from shared memory
        out[transposed_offset] = tile[irow][icol];
    }
}

__global__ void transposeSmemDyn(float* out, float* in, int nrows, int ncols) {
    // dynamic shared memory
    extern __shared__ float tile[];

    // coordinate in original matrix
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

    // linear global memory index for original matrix
    unsigned int offset = INDEX(row, col, ncols);

    // thread index in transposed block
    unsigned int row_idx, col_idx, irow, icol;
    row_idx = threadIdx.y * blockDim.x + threadIdx.x;
    irow = row_idx / blockDim.y;
    icol = row_idx % blockDim.y;
    col_idx = irow * blockDim.x + icol;

    // linear global memory index for transposed matrix
    unsigned int transposed_offset = INDEX(col, row, nrows);

    // transpose with boundary test
    if (row < nrows && col < ncols) {
        // load data from global memory to shared memory
        tile[row_idx] = in[offset];

        // thread synchronization
        __syncthreads();

        // store data to global memory from shared memory
        out[transposed_offset] = tile[col_idx];
    }
}

__global__ void transposeSmemPadDyn(float* out, float* in, int nrows, int ncols) {
    // static shared memory with padding
    extern __shared__ float tile[];

    // coordinate in original matrix
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

    // linear global memory index for original matrix
    unsigned int offset = INDEX(row, col, ncols);

    // thread index in transposed block
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;
    unsigned int col_idx = irow * (blockDim.x + IPAD) + icol;

    // linear global memory index for transposed matrix
    unsigned int transposed_offset = INDEX(col, row, nrows);

    // transpose with boundary test
    if (row < nrows && col < ncols) {
        // load data from global memory to shared memory
        tile[row_idx] = in[offset];

        // thread synchronization
        __syncthreads();

        // store data to global memory from shared memory
        out[transposed_offset] = tile[col_idx];
    }
}

int main(int argc, char** argv) {
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("[host] %s starting transpose at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    bool iprint = 0;

    // set up array size 2048
    int nrows = 1 << 12;
    int ncols = 1 << 12;

    if (argc > 1) iprint = atoi(argv[1]);
    if (argc > 2) nrows = atoi(argv[2]);
    if (argc > 3) ncols = atoi(argv[3]);

    printf(" with matrix nrows %d ncols %d\n", nrows, ncols);
    size_t ncells = nrows * ncols;
    size_t nBytes = ncells * sizeof(float);

    // execution configuration
    dim3 block(BDIMX, BDIMY);
    /*
     * Map CUDA blocks/threads to output space. Map rows in output to same
     * x-value in CUDA, columns to same y-value.
     */
    dim3 grid((ncols + block.x - 1) / block.x, (nrows + block.y - 1) / block.y);
    dim3 grid2((grid.x + 2 - 1) / 2, grid.y);

    // allocate host memory
    float* h_A = (float*)malloc(nBytes);
    float* hostRef = (float*)malloc(nBytes);
    float* gpuRef = (float*)malloc(nBytes);

    //  initialize host array
    initialData(h_A, nrows * ncols);

    //  transpose at host side
    transposeHost(hostRef, h_A, nrows, ncols);

    // allocate device memory
    float* d_A, * d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    // tranpose gmem
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);
    copyGmem << <grid, block >> > (d_C, d_A, nrows, ncols);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    if (iprint) printData(gpuRef, nrows * ncols);

    // tranpose gmem
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);
    naiveGmem << <grid, block >> > (d_C, d_A, nrows, ncols);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    if (iprint) printData(gpuRef, ncells);
    checkResult(hostRef, gpuRef, ncols, nrows);

    // tranpose smem
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);
    naiveGmemUnroll << <grid2, block >> > (d_C, d_A, nrows, ncols);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    if (iprint) printData(gpuRef, ncells);
    checkResult(hostRef, gpuRef, ncols, nrows);

    // tranpose smem
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);
    transposeSmem << <grid, block >> > (d_C, d_A, nrows, ncols);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    if (iprint) printData(gpuRef, ncells);
    checkResult(hostRef, gpuRef, ncols, nrows);

    // tranpose smem pad
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);
    transposeSmemPad << <grid, block >> > (d_C, d_A, nrows, ncols);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    if (iprint) printData(gpuRef, ncells);
    checkResult(hostRef, gpuRef, ncols, nrows);

    // tranpose smem pad
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);
    transposeSmemDyn << <grid, block, BDIMX* BDIMY * sizeof(float) >> > (d_C, d_A, nrows, ncols);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    if (iprint) printData(gpuRef, ncells);
    checkResult(hostRef, gpuRef, ncols, nrows);

    // tranpose smem pad
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);
    transposeSmemPadDyn << <grid, block, (BDIMX + IPAD)* BDIMY * sizeof(float) >> > (d_C, d_A, nrows, ncols);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    if (iprint) printData(gpuRef, ncells);
    checkResult(hostRef, gpuRef, ncols, nrows);

    // tranpose smem
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);
    transposeSmemUnroll << <grid2, block >> > (d_C, d_A, nrows, ncols);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    if (iprint) printData(gpuRef, ncells);
    checkResult(hostRef, gpuRef, ncols, nrows);

    // tranpose smem
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);
    transposeSmemUnrollPad << <grid2, block >> > (d_C, d_A, nrows, ncols);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    if (iprint) printData(gpuRef, ncells);
    checkResult(hostRef, gpuRef, ncols, nrows);

    // tranpose smem
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);
    transposeSmemUnrollPadDyn << <grid2, block, (BDIMX * 2 + IPAD)* BDIMY *sizeof(float) >> > (d_C, d_A, nrows, ncols);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    if (iprint) printData(gpuRef, ncells);
    checkResult(hostRef, gpuRef, ncols, nrows);

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
c:\coding\Cuda\x64\Debug>nvprof ./Cuda.exe
==29428== NVPROF is profiling process 29428, command: ./Cuda.exe
[host] ./Cuda.exe starting transpose at device 0: NVIDIA GeForce MX450  with matrix nrows 4096 ncols 4096
==29428== Profiling application: ./Cuda.exe
==29428== Warning: 20 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==29428== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   72.71%  213.14ms        10  21.314ms  20.885ms  23.029ms  [CUDA memcpy DtoH]
                    7.41%  21.731ms         1  21.731ms  21.731ms  21.731ms  [CUDA memcpy HtoD]
                    4.47%  13.094ms        10  1.3094ms  1.3087ms  1.3109ms  [CUDA memset]
                    1.77%  5.1915ms         1  5.1915ms  5.1915ms  5.1915ms  transposeSmemPad(float*, float*, int, int)
                    1.69%  4.9650ms         1  4.9650ms  4.9650ms  4.9650ms  transposeSmem(float*, float*, int, int)
                    1.67%  4.9008ms         1  4.9008ms  4.9008ms  4.9008ms  naiveGmemUnroll(float*, float*, int, int)
                    1.59%  4.6640ms         1  4.6640ms  4.6640ms  4.6640ms  copyGmem(float*, float*, int, int)
                    1.58%  4.6328ms         1  4.6328ms  4.6328ms  4.6328ms  naiveGmem(float*, float*, int, int)
                    1.55%  4.5552ms         1  4.5552ms  4.5552ms  4.5552ms  transposeSmemPadDyn(float*, float*, int, int)
                    1.51%  4.4212ms         1  4.4212ms  4.4212ms  4.4212ms  transposeSmemDyn(float*, float*, int, int)
                    1.48%  4.3278ms         1  4.3278ms  4.3278ms  4.3278ms  transposeSmemUnrollPad(float*, float*, int, int)
                    1.30%  3.8037ms         1  3.8037ms  3.8037ms  3.8037ms  transposeSmemUnroll(float*, float*, int, int)
                    1.26%  3.6989ms         1  3.6989ms  3.6989ms  3.6989ms  transposeSmemUnrollPadDyn(float*, float*, int, int)
      API calls:   50.05%  243.36ms        11  22.124ms  21.278ms  25.725ms  cudaMemcpy
                   24.97%  121.41ms         1  121.41ms  121.41ms  121.41ms  cudaSetDevice
                    9.30%  45.205ms        10  4.5205ms  3.7039ms  5.1967ms  cudaDeviceSynchronize
                    8.28%  40.243ms        10  4.0242ms  52.400us  39.557ms  cudaLaunchKernel
                    6.89%  33.527ms         1  33.527ms  33.527ms  33.527ms  cudaDeviceReset
                    0.25%  1.2047ms         2  602.35us  300.40us  904.30us  cudaFree
                    0.14%  675.30us        10  67.530us  36.800us  91.800us  cudaMemset
                    0.10%  509.40us         2  254.70us  167.20us  342.20us  cudaMalloc
                    0.01%  69.400us         1  69.400us  69.400us  69.400us  cuLibraryUnload
                    0.01%  47.600us       114     417ns       0ns  27.600us  cuDeviceGetAttribute
                    0.00%  10.500us         1  10.500us  10.500us  10.500us  cudaGetDeviceProperties
                    0.00%  8.1000us         3  2.7000us     100ns  7.8000us  cuDeviceGetCount
                    0.00%  2.5000us         1  2.5000us  2.5000us  2.5000us  cuModuleGetLoadingMode
                    0.00%  2.3000us         1  2.3000us  2.3000us  2.3000us  cuDeviceTotalMem
                    0.00%  1.1000us         2     550ns     100ns  1.0000us  cuDeviceGet
                    0.00%  1.1000us         1  1.1000us  1.1000us  1.1000us  cuDeviceGetName
                    0.00%     800ns         1     800ns     800ns     800ns  cuDeviceGetLuid
                    0.00%     100ns         1     100ns     100ns     100ns  cuDeviceGetUuid
*/
