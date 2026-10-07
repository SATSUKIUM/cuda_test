#include <cuda_runtime.h>
#include "cuda_compat.h" // In this code, cuda::ceil_div()
#include <stdio.h>
#include <iostream>

#include <chrono>
#include <iomanip>

#include <string>
#include <sstream>

#include <vector>

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

void initArray(float* arra, int length, float value)
{
    for(int i = 0; i < length; i++){
        arra[i] = value;
    }
} // void initArray(float* arra, int length, float value)

void vecAddCPU(float* A, float* B, float* C, int length)
{
    for(int i = 0; i < length; i++){
        C[i] = A[i] + B[i];
    }
} // void vecAddCPU(float* A, float* B, float* C, int length)

int main(int argc, char* argv[]){
    // linear additon of two vectors A and B of size 65536, store the result in vector C
    const int length = 1u << 16; // 65536

    // set the number of threads and blocks
    const int numThreads = 256;
    const int numBlocks = cuda_compat::ceil_div(length, numThreads);
    std::cout << "numBlocks: " << numBlocks << std::endl;
    std::cout << "numThreads: " << numThreads << std::endl;

    // warm-up the GPU
    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();
    doWarmUp<<<1, 1>>>();
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double warmUpTime = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Warm-up time: " << std::fixed << std::setprecision(2) << warmUpTime << " ms" << std::endl;


    // declare
    float* A;
    float* B;
    float* C;
    float* C_pu;

    float* devA;
    float* devB;
    float* devC;

    // memory allocation
    cudaMallocHost(&A, length * sizeof(float));
    cudaMallocHost(&B, length * sizeof(float));
    cudaMallocHost(&C, length * sizeof(float));
    C_pu = (float*)malloc(length * sizeof(float));

    cudaMalloc(&devA, length * sizeof(float));
    cudaMalloc(&devB, length * sizeof(float));
    cudaMalloc(&devC, length * sizeof(float));

    // initialize seed arrays
    initArray(A, length, 1.0f);
    initArray(B, length, 2.0f);

    // copy data from host to device
    cudaMemcpy(devA, A, length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, length * sizeof(float), cudaMemcpyHostToDevice);

    // kernel execution
    std::chrono::high_resolution_clock::time_point kernelStart, kernelEnd;
    kernelStart = std::chrono::high_resolution_clock::now();
    vecAdd<<<numBlocks, numThreads>>>(A, B, C, length);
    cudaDeviceSynchronize();
    kernelEnd = std::chrono::high_resolution_clock::now();
    double kernelTime = std::chrono::duration<double, std::milli>(kernelEnd - kernelStart).count();
    std::cout << "Kernel execution time: " << std::fixed << std::setprecision(2) << kernelTime << " ms" << std::endl;

    // copy data from device to host
    cudaMemcpy(C, devC, length * sizeof(float), cudaMemcpyDeviceToHost);

    // verify the result
    vecAddCPU(A, B, C_pu, length);
    for(int i = 0; i < length; i++){
        if(C[i] != 3.0f){
            std::cout << "Error: C[" << i << "] = " << C[i] << std::endl;
            return 1;
        }
    }
    std::cout << "Success: All values in C are correct." << std::endl;
    return 0;
} // int main(int argc, char* argv)