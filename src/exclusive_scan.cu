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

#include <thrust/scan.h> // for thrust::exclusive_scan
#include <thrust/device_ptr.h> // for thrust::device_ptr
#include <thrust/execution_policy.h> // for thrust::device

#include <bitset>


__global__ void doWarmUp()
{
    // do "warm up", and nothing else
    return;
} // __global__ void doWarmUp()

__global__ void collectResults(uint32_t* summedPrefix, uint32_t* original, uint32_t* result, int length)
{
    int workerId = blockDim.x * blockIdx.x + threadIdx.x;

    if(workerId < length){
        if(original[workerId] != 0){
            result[summedPrefix[workerId]] = workerId;
        }
    }
    else{
        // printf("Worker %d is out of bounds for length %d\n", workerId, length);
    }

    return;
} // __global__ void collectResults(uint32_t* summedPrefix, uint32_t* original, uint32_t* result, int length)

void initArray(uint32_t* array_, int length_)
{
    std::vector<uint32_t> initVec(length_);
    initVec = {0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0};
    for(int i = 0; i < length_; i++){
        array_[i] = initVec[i % initVec.size()];
    }
} // void initArray(uint32_t* array_, int length_, uint32_t value_)

int main(int argc, char* argv[]){
    const std::string funcName = "[exclusive_scan.cu/main] ";
    std::cout << funcName << "start" << std::endl;
    // linear additon of two vectors A and B of size 65536, store the result in vector C
    const int length = 1u << 17; // 131072

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
    uint32_t* B;
    cudaMallocHost(&A, length * sizeof(uint32_t));
    cudaMallocHost(&B, length * sizeof(uint32_t));
    initArray(A, length);

    uint32_t* devA;
    uint32_t* devB;
    cudaMalloc(&devA, length * sizeof(uint32_t));
    cudaMalloc(&devB, length * sizeof(uint32_t));
    // copy the array from host to device
    std::chrono::high_resolution_clock::time_point copyStart, copyEnd;
    copyStart = std::chrono::high_resolution_clock::now();
    cudaMemcpy(devA, A, length * sizeof(uint32_t), cudaMemcpyHostToDevice);
    copyEnd = std::chrono::high_resolution_clock::now();
    double copyTime = std::chrono::duration<double, std::micro>(copyEnd - copyStart).count();
    std::cout << "Copy time: " << std::fixed << std::setprecision(2) << copyTime << " us" << std::endl;

    // perform exclusive scan using Thrust
    std::chrono::high_resolution_clock::time_point scanStart, scanEnd;
    scanStart = std::chrono::high_resolution_clock::now();
    thrust::exclusive_scan(thrust::device, thrust::device_pointer_cast(devA), thrust::device_pointer_cast(devA) + length, thrust::device_pointer_cast(devB));
    scanEnd = std::chrono::high_resolution_clock::now();
    double scanTime = std::chrono::duration<double, std::micro>(scanEnd - scanStart).count();
    std::cout << "Scan time: " << std::fixed << std::setprecision(2) << scanTime << " us" << std::endl;

    // copy the result back to host
    std::chrono::high_resolution_clock::time_point copyBackStart, copyBackEnd;
    copyBackStart = std::chrono::high_resolution_clock::now();
    cudaMemcpy(B, devB, length * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    copyBackEnd = std::chrono::high_resolution_clock::now();
    double copyBackTime = std::chrono::duration<double, std::micro>(copyBackEnd - copyBackStart).count();
    std::cout << "Copy back time: " << std::fixed << std::setprecision(2) << copyBackTime << " us" << std::endl;

    // allocate memory for the result
    uint32_t* C;
    cudaMallocHost(&C, (length >> 1) * sizeof(uint32_t));
    uint32_t* devC;
    cudaMalloc(&devC, (length >> 1) * sizeof(uint32_t));
    cudaMemset(devC, 0, (length >> 1) * sizeof(uint32_t));

    // perform atomic addition to collect results
    std::chrono::high_resolution_clock::time_point atomicStart, atomicEnd;
    atomicStart = std::chrono::high_resolution_clock::now();
    collectResults<<<numBlocks, numThreads>>>(devB, devA, devC, length);
    cudaDeviceSynchronize();
    atomicEnd = std::chrono::high_resolution_clock::now();
    double atomicTime = std::chrono::duration<double, std::micro>(atomicEnd - atomicStart).count();
    std::cout << "Atomic addition time: " << std::fixed << std::setprecision(2) << atomicTime << " us" << std::endl;

    // copy the result back to host
    std::chrono::high_resolution_clock::time_point copyResultStart, copyResultEnd;
    copyResultStart = std::chrono::high_resolution_clock::now();
    cudaMemcpy(C, devC, (length >> 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    copyResultEnd = std::chrono::high_resolution_clock::now();
    double copyResultTime = std::chrono::duration<double, std::micro>(copyResultEnd - copyResultStart).count();
    std::cout << "Copy result time: " << std::fixed << std::setprecision(2) << copyResultTime << " us" << std::endl;


    std::cout << "indices: ";
    int looplimit = std::min(length, 32);
    for(int i = 0; (i < looplimit) && (i < length); i++){
        std::cout << std::setw(2) << std::setfill(' ') << i << " ";
    }
    std::cout << std::endl;
    std::cout << "A:       ";
    for(int i = 0; (i < looplimit) && (i < length); i++){
        std::cout << std::setw(2) << std::setfill(' ') << A[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "B:       ";
    for(int i = 0; (i < looplimit) && (i < length); i++){
        std::cout << std::setw(2) << std::setfill(' ') << B[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "C:       ";
    for(int i = 0; (i < looplimit) && (i < length >> 1); i++){
        std::cout << std::setw(2) << std::setfill(' ') << C[i] << " ";
    }
    std::cout << std::endl;

    return 0;
} // int main(int argc, char* argv)