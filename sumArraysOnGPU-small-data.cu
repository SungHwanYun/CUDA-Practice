#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<time.h>
#include<string.h>

// error check
#define CHECK(call) \
{ \
    const cudaError_t error = call;\
    if (error != cadaSuccess) { \
        printf("[device] Error: %s %d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
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
void initialData(float* ip, const int N) {
    // generate different seed for random number
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < N; i++) {
        ip[i] = (float)(rand() & 0xff) / 10.0f;
    }
}

// C[i] = A[i] + B[i], 0<= i <= N-1
void sumArraysOnHost(float* A, float* B, float* C, const int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
    printf("[device] thread %d : %5.2f + %5.2f = %5.2f\n", i, A[i], B[i], C[i]);
}

int main(int argc, char **argv) {
    printf("[host] %s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaSetDevice(dev);

    // set up data size of vectors
    int nElem = 32;
    printf("[host] Vector size %d\n", nElem);

    // malloc host memory
    size_t nBytes = nElem * sizeof(float);
    float* h_A, * h_B, * hostRef, * gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, sizeof(hostRef));
    memset(gpuRef, 0, sizeof(gpuRef));

    // malloc device global memory
    float* d_A, * d_B, * d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    // transfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // invoke kernel at host side
    dim3 block(nElem, 1, 1);
    dim3 grid((nElem + block.x - 1) / block.x, 1, 1);

    sumArraysOnGPU << <grid, block >> > (d_A, d_B, d_C);
    printf("[host] Execution configuration <<<%d %d>>>\n", grid.x, block.x);

    // copy kernel result back to host side
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    // add vector at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
}

/*
output:
[host] C:\coding\CudaRuntime1\x64\Debug\CudaRuntime1.exe Starting...
[host] Vector size 32
[host] Execution configuration <<<1 32>>>
[device] thread 0 :  6.90 +  6.90 = 13.80
[device] thread 1 :  9.90 +  9.90 = 19.80
[device] thread 2 : 14.20 + 14.20 = 28.40
[device] thread 3 : 17.50 + 17.50 = 35.00
[device] thread 4 : 21.10 + 21.10 = 42.20
[device] thread 5 : 12.40 + 12.40 = 24.80
[device] thread 6 :  5.00 +  5.00 = 10.00
[device] thread 7 :  4.70 +  4.70 =  9.40
[device] thread 8 :  7.50 +  7.50 = 15.00
[device] thread 9 : 10.90 + 10.90 = 21.80
[device] thread 10 : 14.30 + 14.30 = 28.60
[device] thread 11 : 19.30 + 19.30 = 38.60
[device] thread 12 : 21.20 + 21.20 = 42.40
[device] thread 13 :  8.40 +  8.40 = 16.80
[device] thread 14 : 15.90 + 15.90 = 31.80
[device] thread 15 : 13.90 + 13.90 = 27.80
[device] thread 16 : 21.00 + 21.00 = 42.00
[device] thread 17 : 14.20 + 14.20 = 28.40
[device] thread 18 : 14.70 + 14.70 = 29.40
[device] thread 19 :  0.20 +  0.20 =  0.40
[device] thread 20 : 17.40 + 17.40 = 34.80
[device] thread 21 : 11.40 + 11.40 = 22.80
[device] thread 22 : 18.90 + 18.90 = 37.80
[device] thread 23 :  7.10 +  7.10 = 14.20
[device] thread 24 : 21.70 + 21.70 = 43.40
[device] thread 25 :  6.30 +  6.30 = 12.60
[device] thread 26 : 17.20 + 17.20 = 34.40
[device] thread 27 : 18.30 + 18.30 = 36.60
[device] thread 28 :  1.40 +  1.40 =  2.80
[device] thread 29 :  8.20 +  8.20 = 16.40
[device] thread 30 : 15.30 + 15.30 = 30.60
[device] thread 31 : 19.00 + 19.00 = 38.00
[host] Arrays match.
*/
