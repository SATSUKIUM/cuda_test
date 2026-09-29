#include <cuda_runtime.h>
#include <iostream>

__global__ void vecAdd(float* A, float* B, float* C, int length)
{
    int workerId = blockDim.x * blockIdx.x + threadIdx.x;

    if(workerId < length){
        C[workerId] = A[workerId] + B[workerId];
    }
} // __global__ void vecAdd(float* A, float* B, float* C, int length)

int main(int argc, char* argv){
    if(argc == 1){
        std::cout << "Usage: \"./bin/test 5 5\" for 5 thread blocks, 5 thread" << std::endl;
    }
    else{
        // linear additon of two vectors A and B of size 65536, store the result in vector C
        const int length = 65536;
        float* A = new float[length];
        float* B = new float[length];
        float* C = new float[length];
        vecAdd<<<4, 256>>>(A, B, C, length);
    }
} // int main(int argc, char* argv)