#include <cuda_runtime.h>
#include <cuda/cmath> // In this code, cuda::ceil_div()

#include <stdio.h>
#include <iostream>

#include <chrono>
#include <iomanip>

#include <string>
#include <sstream>

#include <vector>

#include <TH1.h>
#include <TCanvas.h>
#include <TPad.h>
#include <TStyle.h>
#include <TFile.h>
#include <TGraph.h>

#include <TROOT.h> // for gROOT->SetBatch(kTRUE);

__global__ void vecAdd(float* A, float* B, float* C, int length);
__global__ void doWarmUp();
void initializeVectors(float* A, float* B, int length);
void serialVecAdd(float* A, float* B, float* C_pu, int length);


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

void initializeVectors(float* A, float* B, int length)
{
    for(int i = 0; i < length; i++){
        A[i] = 1.0f;
        B[i] = 2.0f;
    }
} // void initializeVectors(float* A, float* B, int length)

void serialVecAdd(float* A, float* B, float* C_pu, int length)
{
    for(int i = 0; i < length; i++){
        C_pu[i] = A[i] + B[i];
    }
}

int main(int argc, char* argv[]){
    // linear additon of two vectors A and B of size 65536, store the result in vector C
    // using よしなにやってくれるやり方
    const int length = 1u << 24; // 16,777,216 elements (2^24)
    const int nLoops = 1e4; // 10,000 loops
    std::cout << "Vector length: " << length << std::endl;
    std::cout << "Number of loops: " << nLoops << std::endl;

    gROOT->SetBatch(kTRUE); // Disable interactive mode for ROOT

    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();
    doWarmUp<<<1, 1>>>();
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double warmUpTime = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Warm-up time: " << std::fixed << std::setprecision(2) << warmUpTime << " ms" << std::endl;

    const int numThreads = 256; // 256 threads per block
    const int numBlocks = cuda::ceil_div(length, numThreads);
    std::cout << "numBlocks: " << numBlocks << std::endl;
    std::cout << "numThreads: " << numThreads << std::endl;

    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* C_pu = (float*)malloc(length * sizeof(float));

    cudaMallocManaged(&A, length * sizeof(float));
    cudaMallocManaged(&B, length * sizeof(float));
    cudaMallocManaged(&C, length * sizeof(float));

    initializeVectors(A, B, length);

    std::vector<double> kernelTimes;
    kernelTimes.reserve(nLoops); // Reserve space for 500,000 elements

    std::chrono::high_resolution_clock::time_point before_loop = std::chrono::high_resolution_clock::now();
    for(int i=0; i<nLoops; ++i){
        start = std::chrono::high_resolution_clock::now();
        vecAdd<<<numBlocks, numThreads>>>(A, B, C, length);
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        double kernelTime = std::chrono::duration<double, std::micro>(end - start).count();
        kernelTimes.push_back(kernelTime);
    }
    std::chrono::high_resolution_clock::time_point after_loop = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double, std::milli>(after_loop - before_loop).count();
    std::cout << "Total time for " << nLoops << " iterations: " << std::fixed << std::setprecision(2) << totalTime << " ms, -> " << (totalTime / nLoops) << " ms/iteration" << std::endl;

    serialVecAdd(A, B, C_pu, length);

    for(int i = 0; i < length; i++){
        if(C[i] != C_pu[i]){
            std::cout << "Error: C[" << i << "] = " << C[i] << " != " << C_pu[i] << std::endl;
            return 1;
        }
    }
    std::cout << "Success: All values in C are correct." << std::endl;
    return 0;
} // int main(int argc, char* argv)