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

#include "cudamap.h"

namespace cudamap {

bool FEEAddrDecoder::initialize(const std::vector<FEEAddr>& feeAddrArray, const std::vector<TriggerConfig>& configArray)
{
    const std::string funcName = "[cudamap::FEEAddrDecoder::initialize] ";
    std::cout << funcName << "initializing Trigger Configuration Decoder" << std::endl;
    if (feeAddrArray.size() != configArray.size()) {
        std::cerr
            << "Number of items does not match between "
            << "feeAddrArray and configArray"
            << std::endl;
        return false;
    }
    if (feeAddrArray.empty()) {
        return false;
    }

    minIP3rd = UINT8_MAX;
    maxIP3rd = 0;
    minIP4th = UINT8_MAX;
    maxIP4th = 0;
    minCh = UINT8_MAX;
    maxCh = 0;

    // scan max and min values of the keys

    for (const auto& addr : feeAddrArray) {
        if (addr.ip3rd < minIP3rd) minIP3rd = addr.ip3rd;
        if (addr.ip3rd > maxIP3rd) maxIP3rd = addr.ip3rd;

        if (addr.ip4th < minIP4th) minIP4th = addr.ip4th;
        if (addr.ip4th > maxIP4th) maxIP4th = addr.ip4th;

        if (addr.ch < minCh) minCh = addr.ch;
        if (addr.ch > maxCh) maxCh = addr.ch;
    }

    sizeSpaceOfIP3rd = static_cast<uint16_t>(maxIP3rd) - static_cast<uint16_t>(minIP3rd) + 1;
    sizeSpaceOfIP4th = static_cast<uint16_t>(maxIP4th) - static_cast<uint16_t>(minIP4th) + 1;
    sizeSpaceOfCh = static_cast<uint16_t>(maxCh) - static_cast<uint16_t>(minCh) + 1;
    sizeSpaceOfKey = static_cast<uint32_t>(sizeSpaceOfIP3rd) * static_cast<uint32_t>(sizeSpaceOfIP4th) * static_cast<uint32_t>(sizeSpaceOfCh);

    #if 1
    std::cout << "IP3rd: [" << static_cast<int>(minIP3rd) << ", " << static_cast<int>(maxIP3rd) << "]"
              << " IP4th: [" << static_cast<int>(minIP4th) << ", " << static_cast<int>(maxIP4th) << "]"
              << " Ch: [" << static_cast<int>(minCh) << ", " << static_cast<int>(maxCh) << "]"
              << std::endl;
    std::cout << "\tsizeSpaceOfKey = sizeSpaceOfIP3rd * sizeSpaceOfIP4th * sizeSpaceOfCh = "
              << sizeSpaceOfIP3rd << " * "
              << sizeSpaceOfIP4th << " * "
              << sizeSpaceOfCh
              << " = " << sizeSpaceOfKey
              << std::endl;
    std::cout << "\t\t" << static_cast<double>(sizeSpaceOfKey) / static_cast<double>(UINT32_MAX) * 100.0 << " % of UINT32_MAX" << std::endl;
    #endif

    // allocate memory
    triggerConfigArray = static_cast<TriggerConfig*>(std::calloc(sizeSpaceOfKey, sizeof(TriggerConfig)));
    if (triggerConfigArray == nullptr) {
        return false;
    }


    // register
    for (size_t i = 0; i < feeAddrArray.size(); ++i) {
        uint32_t key;
        if (!getKey(feeAddrArray[i].ip3rd, feeAddrArray[i].ip4th, feeAddrArray[i].ch, key)){
            continue;
        }
        triggerConfigArray[key] = configArray[i];
    }

    std::cout << funcName << "Trigger Configuration Decoder initialized with " << sizeSpaceOfKey << " * " << sizeof(TriggerConfig) << " = " << sizeSpaceOfKey * sizeof(TriggerConfig) << " bytes" << std::endl;
    return true;
} // bool FEEAddrDecoder::initialize(const std::vector<FEEAddr>& feeAddrArray, const std::vector<TriggerConfig>& configArray)


__host__ __device__ bool FEEAddrDecoder::getKey(uint8_t ip3rd, uint8_t ip4th, uint8_t ch, uint32_t& key) const
{
    if ((ip3rd < minIP3rd) || (ip3rd > maxIP3rd) || (ip4th < minIP4th) || (ip4th > maxIP4th) || (ch < minCh) || (ch > maxCh)) {
        return false;
    }

    key = (static_cast<uint32_t>(ip3rd) - minIP3rd) * sizeSpaceOfIP4th * sizeSpaceOfCh + (static_cast<uint32_t>(ip4th) - minIP4th) * sizeSpaceOfCh + (static_cast<uint32_t>(ch) - minCh);
    return true;
} // __host__ __device__ bool FEEAddrDecoder::getKey(uint8_t ip3rd, uint8_t ip4th, uint8_t ch, uint32_t& key) const


__host__ __device__ TriggerConfig FEEAddrDecoder::getTriggerConfig(uint32_t key) const
{
    if (key >= sizeSpaceOfKey) {
        return TriggerConfig{};
    }

    return triggerConfigArray[key];
} // __host__ __device__ TriggerConfig FEEAddrDecoder::getTriggerConfig(uint32_t key) const

} // namespace cudamap

__global__ void testDecoder(const cudamap::FEEAddrDecoder* decoder)
{
    printf("testDecoder called\n");
    uint32_t key;

    if (decoder->getKey(0x02, 0xa0, 3, key)) { // for 192.168.2.160, ch 3
        auto config = decoder->getTriggerConfig(key);
        printf("key=%u iSubTimeRegion=%u hitBit=%u delay=%u width=%u\n", key, config.iSubTimeRegion, config.hitBit, config.delay, config.width);
    }
    if(decoder->getKey(0x02, 0xa1, 6, key)) { // for 192.168.2.161, ch 6
        auto config = decoder->getTriggerConfig(key);
        printf("key=%u iSubTimeRegion=%u hitBit=%u delay=%u width=%u\n", key, config.iSubTimeRegion, config.hitBit, config.delay, config.width);
    }
} // __global__ void testDecoder(const cudamap::FEEAddrDecoder* decoder)




__global__ void vecAdd(float* A, float* B, float* C, int length)
{
    int workerId = blockDim.x * blockIdx.x + threadIdx.x;

    if(workerId < length){
        C[workerId] = A[workerId] + B[workerId];
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
}

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


    // kernel execution
    std::chrono::high_resolution_clock::time_point kernelStart, kernelEnd;
    kernelStart = std::chrono::high_resolution_clock::now();
    doWarmUp<<<numBlocks, numThreads>>>();
    cudaDeviceSynchronize();
    kernelEnd = std::chrono::high_resolution_clock::now();
    double kernelTime = std::chrono::duration<double, std::milli>(kernelEnd - kernelStart).count();
    std::cout << "Kernel execution time: " << std::fixed << std::setprecision(2) << kernelTime << " ms" << std::endl;

    // test cudamap
    cudamap::FEEAddrDecoder decoder;
    std::vector<cudamap::FEEAddr> feeAddrs;
    std::vector<cudamap::TriggerConfig> triggerConfigs;

    // Add some test data
    cudamap::FEEAddr addr1{0x02, 0xa0, 3}; // for 192.168.2.160, channel 3
    cudamap::FEEAddr addr2{0x02, 0xa1, 6}; // for 192.168.2.161, channel 6
    feeAddrs.push_back(addr1);
    feeAddrs.push_back(addr2);

    cudamap::TriggerConfig config1{0, 0, 0, 20};
    cudamap::TriggerConfig config2{UINT8_MAX, 1, 1, 20};
    triggerConfigs.push_back(config1);
    triggerConfigs.push_back(config2);

    decoder.initialize(feeAddrs, triggerConfigs);

    int nCongis = decoder.getSizeSpaceOfKey();
    cudamap::TriggerConfig* devTriggerConfigs;
    cudaMalloc(&devTriggerConfigs, nCongis * sizeof(cudamap::TriggerConfig));
    cudaMemcpy(devTriggerConfigs, decoder.getTriggerConfigArray(), nCongis * sizeof(cudamap::TriggerConfig), cudaMemcpyHostToDevice);

    cudamap::FEEAddrDecoder devDecoder = decoder;
    devDecoder.setTriggerConfigArray(devTriggerConfigs);

    cudamap::FEEAddrDecoder* devDecoderPtr;
    cudaMalloc(&devDecoderPtr, sizeof(cudamap::FEEAddrDecoder));
    cudaMemcpy(devDecoderPtr, &devDecoder, sizeof(cudamap::FEEAddrDecoder), cudaMemcpyHostToDevice);

    std::cout << "before testDecoder kernel launch" << std::endl;

    testDecoder<<<1, 1>>>(devDecoderPtr);

    // launch時のエラー
    cudaError_t err = cudaGetLastError();
    std::cout << "launch: "
            << cudaGetErrorString(err)
            << std::endl;

    // 実行時のエラー
    err = cudaDeviceSynchronize();
    std::cout << "sync: "
            << cudaGetErrorString(err)
            << std::endl;

    std::cout << "after testDecoder kernel" << std::endl;


    return 0;
} // int main(int argc, char* argv)