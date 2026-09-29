#include <cuda_runtime.h>
#include <cuda/cmath> // In this code, cuda::ceil_div()
#include <stdio.h>
#include <iostream>

#include <chrono>
#include <iomanip>

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

__global__ void doWarmUp()
{
    // do "warm up", and nothing else
}

int main(int argc, char* argv[]){
    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();
    doWarmUp<<<1, 1>>>();
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double warmUpTime = std::chrono::duration<double, std::micro>(end - start).count();
    std::cout << "Warm-up time: " << std::fixed << std::setprecision(2) << warmUpTime << " microseconds" << std::endl;

    if(argc == 1){
        std::cout << "Usage: \"./bin/test 5 5\" for 5 thread blocks, 5 thread" << std::endl;
        return 1;
    }
    else{
        int arg1 = std::stoi(argv[1]);
        std::cout << "argument 1: " << arg1 << std::endl;

        // linear additon of two vectors A and B of size 65536, store the result in vector C
        const int length = 65536;
        float* A = new float[length];
        float* B = new float[length];
        float* C = new float[length];

        for(int i = 0; i < length; i++){
            A[i] = 1.0f;
            B[i] = 2.0f;
        }

        const int numThreads = std::stoi(argv[1]);
        const int numBlocks = cuda::ceil_div(length, numThreads);
        std::cout << "numBlocks: " << numBlocks << std::endl;
        std::cout << "numThreads: " << numThreads << std::endl;

        start = std::chrono::high_resolution_clock::now();
        vecAdd<<<numBlocks, numThreads>>>(A, B, C, length);
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        double kernelTime = std::chrono::duration<double, std::micro>(end - start).count();
        std::cout << "Kernel execution time: " << std::fixed << std::setprecision(2) << kernelTime << " microseconds" << std::endl;

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