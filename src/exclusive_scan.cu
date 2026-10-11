#include <cuda_runtime.h>
#include "cuda_compat.h" // In this code, cuda::ceil_div()
#include <stdio.h>
#include <iostream>

#include <chrono>
#include <iomanip>

#include <string>
#include <sstream>

#include <vector>

#include <cstdlib>
#include <iostream>
#include <limits>

#include <thrust/scan.h>

#include <bitset>


__global__ void vecAdd(uint32_t* A, int length)
{
    int workerId = blockDim.x * blockIdx.x + threadIdx.x;

    if(workerId < length){
        // atomicOr(&A[workerId], 1u << workerId);
        atomicOr(&A[0], 1u << workerId);
    }
    else{
        // printf("Worker %d is out of bounds for length %d\n", workerId, length);
    }

    return;
} // __global__ void vecAdd(float* A, float* B, float* C, int length)

__global__ void doWarmUp()
{
    // do "warm up", and nothing else
    return;
} // __global__ void doWarmUp()

void initArray(uint32_t* array_, int length_, uint32_t value_)
{
    for(int i = 0; i < length_; i++){
        array_[i] = value_;
    }
} // void initArray(uint32_t* array_, int length_, uint32_t value_)

int main(int argc, char* argv[]){
    // linear additon of two vectors A and B of size 65536, store the result in vector C
    const int length = 1u << 4; // 16

    // set the number of threads and blocks
    const int numThreads = 256;
    const int numBlocks = cuda_compat::ceil_div(length, numThreads);
    std::cout << "numBlocks: " << numBlocks << std::endl;
    std::cout << "numThreads: " << numThreads << std::endl;

    // warm-up the GPU
    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();
    doWarmUp<<<numBlocks, numThreads>>>();
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double warmUpTime = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Warm-up time: " << std::fixed << std::setprecision(2) << warmUpTime << " ms" << std::endl;

    // declare and initialize the array
    uint32_t* A;
    cudaMallocHost(&A, length * sizeof(uint32_t));
    initArray(A, length, 0);

    uint32_t* devA;
    cudaMalloc(&devA, length * sizeof(uint32_t));
    cudaMemcpy(devA, A, length * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // kernel execution
    std::chrono::high_resolution_clock::time_point kernelStart, kernelEnd;
    kernelStart = std::chrono::high_resolution_clock::now();
    vecAdd<<<numBlocks, numThreads>>>(devA, length);
    cudaDeviceSynchronize();
    kernelEnd = std::chrono::high_resolution_clock::now();
    double kernelTime = std::chrono::duration<double, std::micro>(kernelEnd - kernelStart).count();
    std::cout << "Kernel execution time: " << std::fixed << std::setprecision(2) << kernelTime << " us" << std::endl;

    cudaMemcpy(A, devA, length * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < length; i++){
    //     std::cout << "A[" << i << "] = " << std::bitset<32>(A[i]) << std::endl;
    // }
    std::cout << "A[0] = " << std::bitset<32>(A[0]) << std::endl;
    return 0;
} // int main(int argc, char* argv)