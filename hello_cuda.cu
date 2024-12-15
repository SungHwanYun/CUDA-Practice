#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void helloCUDA() {
    int i = threadIdx.x;
    printf("Hello CUDA from GPU!(threadIdx.x = %d)\n", i);
}

int main() {
    printf("Hello GPU from CPU!\n");
    helloCUDA <<<1, 10>>> ();
    return 0;
}

/*
output:
Hello GPU from CPU!
Hello CUDA from GPU!(threadIdx.x = 0)
Hello CUDA from GPU!(threadIdx.x = 1)
Hello CUDA from GPU!(threadIdx.x = 2)
Hello CUDA from GPU!(threadIdx.x = 3)
Hello CUDA from GPU!(threadIdx.x = 4)
Hello CUDA from GPU!(threadIdx.x = 5)
Hello CUDA from GPU!(threadIdx.x = 6)
Hello CUDA from GPU!(threadIdx.x = 7)
Hello CUDA from GPU!(threadIdx.x = 8)
Hello CUDA from GPU!(threadIdx.x = 9)
*/
