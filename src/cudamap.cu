#include <cuda_runtime.h>
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

// forward declaration of the kernel function
__global__ void doWarmUp();

namespace cudamap {

    bool FEEAddrDecoder::initialize(const std::vector<FEEAddr>& feeAddrArray, const std::vector<TriggerConfig>& configArray)
    {
        const std::string funcName = "[cudamap::FEEAddrDecoder::initialize] ";
        std::cout << funcName << "initializing Trigger Configuration Decoder" << std::endl;
        // out of range check
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

        minIP1st = UINT8_MAX;
        maxIP1st = 0;
        minIP2nd = UINT8_MAX;
        maxIP2nd = 0;
        minIP3rd = UINT8_MAX;
        maxIP3rd = 0;
        minIP4th = UINT8_MAX;
        maxIP4th = 0;
        minCh = UINT8_MAX;
        maxCh = 0;

        // scan max and min values of the keys
        for (const auto& addr : feeAddrArray) {
            if (addr.ip1st < minIP1st){ minIP1st = addr.ip1st; }
            if (addr.ip1st > maxIP1st){ maxIP1st = addr.ip1st; }

            if (addr.ip2nd < minIP2nd){ minIP2nd = addr.ip2nd; }
            if (addr.ip2nd > maxIP2nd){ maxIP2nd = addr.ip2nd; }

            if (addr.ip3rd < minIP3rd){ minIP3rd = addr.ip3rd; }
            if (addr.ip3rd > maxIP3rd){ maxIP3rd = addr.ip3rd; }

            if (addr.ip4th < minIP4th){ minIP4th = addr.ip4th; }
            if (addr.ip4th > maxIP4th){ maxIP4th = addr.ip4th; }

            if (addr.ch < minCh){ minCh = addr.ch; }
            if (addr.ch > maxCh){ maxCh = addr.ch; }
        }

        sizeSpaceOfIP1st = static_cast<uint16_t>(maxIP1st) - static_cast<uint16_t>(minIP1st) + 1;
        sizeSpaceOfIP2nd = static_cast<uint16_t>(maxIP2nd) - static_cast<uint16_t>(minIP2nd) + 1;
        sizeSpaceOfIP3rd = static_cast<uint16_t>(maxIP3rd) - static_cast<uint16_t>(minIP3rd) + 1;
        sizeSpaceOfIP4th = static_cast<uint16_t>(maxIP4th) - static_cast<uint16_t>(minIP4th) + 1;
        sizeSpaceOfCh = static_cast<uint16_t>(maxCh) - static_cast<uint16_t>(minCh) + 1;
        sizeSpaceOfKey = static_cast<uint32_t>(sizeSpaceOfIP1st) * static_cast<uint32_t>(sizeSpaceOfIP2nd) * static_cast<uint32_t>(sizeSpaceOfIP3rd) * static_cast<uint32_t>(sizeSpaceOfIP4th) * static_cast<uint32_t>(sizeSpaceOfCh);

        strideCh = 1;
        strideIP4th = sizeSpaceOfCh;
        strideIP3rd = sizeSpaceOfCh * sizeSpaceOfIP4th;
        strideIP2nd = sizeSpaceOfCh * sizeSpaceOfIP4th * sizeSpaceOfIP3rd;
        strideIP1st = sizeSpaceOfCh * sizeSpaceOfIP4th * sizeSpaceOfIP3rd * sizeSpaceOfIP2nd;

        #if 1
        std::cout << funcName << "min and max values of the keys" << std::endl;
        std::cout << "IP1st: [" << static_cast<int>(minIP1st) << ", " << static_cast<int>(maxIP1st) << "]"
                << " IP2nd: [" << static_cast<int>(minIP2nd) << ", " << static_cast<int>(maxIP2nd) << "]"
                << " IP3rd: [" << static_cast<int>(minIP3rd) << ", " << static_cast<int>(maxIP3rd) << "]"
                << " IP4th: [" << static_cast<int>(minIP4th) << ", " << static_cast<int>(maxIP4th) << "]"
                << " Ch: [" << static_cast<int>(minCh) << ", " << static_cast<int>(maxCh) << "]"
                << std::endl;
        std::cout << "\tsizeSpaceOfKey = sizeSpaceOfIP1st * sizeSpaceOfIP2nd * sizeSpaceOfIP3rd * sizeSpaceOfIP4th * sizeSpaceOfCh = "
                << sizeSpaceOfIP1st << " * "
                << sizeSpaceOfIP2nd << " * "
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
            if (!getKey(feeAddrArray[i].ip1st, feeAddrArray[i].ip2nd, feeAddrArray[i].ip3rd, feeAddrArray[i].ip4th, feeAddrArray[i].ch, key)){
                continue;
            }
            triggerConfigArray[key] = configArray[i];
        }

        std::cout << funcName << "Trigger Configuration Decoder initialized with " << sizeSpaceOfKey << " * " << sizeof(TriggerConfig) << " = " << sizeSpaceOfKey * sizeof(TriggerConfig) << " bytes" << std::endl;
        return true;
    } // bool FEEAddrDecoder::initialize(const std::vector<FEEAddr>& feeAddrArray, const std::vector<TriggerConfig>& configArray)



    __host__ __device__ bool FEEAddrDecoder::getKey(uint8_t ip1st, uint8_t ip2nd, uint8_t ip3rd, uint8_t ip4th, uint8_t ch, uint32_t& key) const
    {
        if ((ip1st < minIP1st) || (ip1st > maxIP1st) || (ip2nd < minIP2nd) || (ip2nd > maxIP2nd) || (ip3rd < minIP3rd) || (ip3rd > maxIP3rd) || (ip4th < minIP4th) || (ip4th > maxIP4th) || (ch < minCh) || (ch > maxCh)) {
            return false;
        }

        key = (static_cast<uint32_t>(ip1st) - minIP1st) * strideIP1st + (static_cast<uint32_t>(ip2nd) - minIP2nd) * strideIP2nd + (static_cast<uint32_t>(ip3rd) - minIP3rd) * strideIP3rd + (static_cast<uint32_t>(ip4th) - minIP4th) * strideIP4th + (static_cast<uint32_t>(ch) - minCh) * strideCh;
        return true;
    } // __host__ __device__ bool FEEAddrDecoder::getKey(uint8_t ip1st, uint8_t ip2nd, uint8_t ip3rd, uint8_t ip4th, uint8_t ch, uint32_t& key) const


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

    // test getKey and getTriggerConfig
    if (decoder->getKey(0xc0, 0xa8, 0x02, 0xa9, 8, key)) { // for 192.168.2.169, ch 8
        printf("%d, %d, %d, %d, %d -> key=%u\n", 0xc0, 0xa8, 0x02, 0xa9, 8, key);
        auto config = decoder->getTriggerConfig(key);
        printf("\tkey=%u: iSubTimeRegion=%u hitBit=%u delay=%u width=%u\n", key, config.iSubTimeRegion, config.hitBit, config.delay, config.width);
    }
    else {
        printf("Failed to get key for 192.168.2.169, ch 8\n");
    }

    if(decoder->getKey(0xc0, 0xa8, 0x02, 0xaa, 16, key)) { // for 192.168.2.170, ch 16
        printf("%d, %d, %d, %d, %d -> key=%u\n", 0xc0, 0xa8, 0x02, 0xaa, 16, key);
        auto config = decoder->getTriggerConfig(key);
        printf("\tkey=%u: iSubTimeRegion=%u hitBit=%u delay=%d width=%u\n", key, config.iSubTimeRegion, config.hitBit, config.delay, config.width);
    }
    else {
        printf("Failed to get key for 192.168.2.170, ch 16\n");
    }
} // __global__ void testDecoder(const cudamap::FEEAddrDecoder* decoder)


int main(int argc, char* argv[]){
    const std::string funcName = "[cudamap.cu/main] ";
    std::cout << funcName << "start" << std::endl;
    // warm-up the GPU
    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();
    doWarmUp<<<1, 1>>>();
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double warmUpTime = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Warm-up time: " << std::fixed << std::setprecision(2) << warmUpTime << " ms" << std::endl;

    // test cudamap
    cudamap::FEEAddrDecoder decoder;
    std::vector<cudamap::FEEAddr> feeAddrs;
    std::vector<cudamap::TriggerConfig> triggerConfigs;

    // Add some test data
    auto add = [&](uint32_t ip, uint8_t ch,
                uint8_t group, uint8_t subgroup,
                int16_t offset) {
        feeAddrs.push_back({static_cast<uint8_t>((ip >> 24) & 0xff), static_cast<uint8_t>((ip >> 16) & 0xff), static_cast<uint8_t>((ip >> 8) & 0xff), static_cast<uint8_t>(ip & 0xff), ch});
        triggerConfigs.push_back({group, subgroup, offset, 20});
    };

    constexpr uint8_t NO_SUBGROUP = UINT8_MAX;

    // Group 0, 1
    add(0xc0a802a9,  static_cast<uint8_t>(8), static_cast<uint8_t>(0), static_cast<uint8_t>(NO_SUBGROUP), static_cast<short>(0));
    add(0xc0a802a9, static_cast<uint8_t>(10), static_cast<uint8_t>(1), static_cast<uint8_t>(NO_SUBGROUP), static_cast<short>(0));
    // Group 2, Subgroup 0
    add(0xc0a802aa, static_cast<uint8_t>(16), static_cast<uint8_t>(2), static_cast<uint8_t>(0), static_cast<short>(-12));
    add(0xc0a802aa, static_cast<uint8_t>(17), static_cast<uint8_t>(2), static_cast<uint8_t>(0), static_cast<short>(-12));
    // Group 2, Subgroup 1
    add(0xc0a802aa, static_cast<uint8_t>(18), static_cast<uint8_t>(2), static_cast<uint8_t>(1), static_cast<short>(-12));
    add(0xc0a802aa, static_cast<uint8_t>(19), static_cast<uint8_t>(2), static_cast<uint8_t>(1), static_cast<short>(-12));
    // Group 2, Subgroup 2
    add(0xc0a802aa, static_cast<uint8_t>(20), static_cast<uint8_t>(2), static_cast<uint8_t>(2), static_cast<short>(-12));
    add(0xc0a802aa, static_cast<uint8_t>(21), static_cast<uint8_t>(2), static_cast<uint8_t>(2), static_cast<short>(-12));
    // Group 2, Subgroup 3
    add(0xc0a802aa, static_cast<uint8_t>(22), static_cast<uint8_t>(2), static_cast<uint8_t>(3), static_cast<short>(-12));
    add(0xc0a802aa, static_cast<uint8_t>(23), static_cast<uint8_t>(2), static_cast<uint8_t>(3), static_cast<short>(-12));
    // Group 2, Subgroup 4
    add(0xc0a802aa, static_cast<uint8_t>(24), static_cast<uint8_t>(2), static_cast<uint8_t>(4), static_cast<short>(-12));
    add(0xc0a802aa, static_cast<uint8_t>(25), static_cast<uint8_t>(2), static_cast<uint8_t>(4), static_cast<short>(-12));
    // Group 2, Subgroup 5
    add(0xc0a802aa, static_cast<uint8_t>(27), static_cast<uint8_t>(2), static_cast<uint8_t>(5), static_cast<short>(-12));
    add(0xc0a802aa, static_cast<uint8_t>(28), static_cast<uint8_t>(2), static_cast<uint8_t>(5), static_cast<short>(-12));

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

__global__ void doWarmUp()
{
    // do "warm up", and nothing else
    return;
}