#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

__global__ void vecAdd(float* A, float* B, float* C, int length)
{
    int workerId = blockDim.x * blockIdx.x + threadIdx.x;

    if(workerId < length){
        C[workerId] = A[workerId] + B[workerId];
    }
    else{
        // printf("Worker %d is out of bounds for length %d\n", workerId, length);
    }
} // __global__ void vecAdd(float* A, float* B, float* C, int length)

int main(int argc, char* argv[]){
    if(argc == 1){
        std::cout << "Usage: \"./bin/test 5 5\" for 5 thread blocks, 5 thread" << std::endl;
        return 1;
    }
    else{
        // linear additon of two vectors A and B of size 65536, store the result in vector C
        const int length = 65536;
        float* A = new float[length];
        float* B = new float[length];
        float* C = new float[length];

        for(int i = 0; i < length; i++){
            A[i] = 1.0f;
            B[i] = 2.0f;
        }

        vecAdd<<<128, 1024>>>(A, B, C, length);
        cudaDeviceSynchronize();

        for(int i = 0; i < length; i++){
            if(C[i] != 3.0f){
                std::cout << "Error: C[" << i << "] = " << C[i] << std::endl;
                return 1;
            }
        }
        std::cout << "Success: All values in C are correct." << std::endl;
        return 0;
    }
} // int main(int argc, char* argv)