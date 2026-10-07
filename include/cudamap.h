#ifndef CUDAMAP_H
#define CUDAMAP_H

#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

namespace cudamap {

struct TriggerConfig
{
    uint8_t  iSubTimeRegion{};
    uint32_t hitBit{};
    uint8_t  delay{};
    uint8_t  width{};
}; // struct TriggerConfig

struct FEEAddr
{
    uint8_t ip3rd{};
    uint8_t ip4th{};
    uint8_t ch{};
}; // struct FEEAddr


class FEEAddrDecoder
{
public:
    FEEAddrDecoder() = default;

    bool initialize(const std::vector<FEEAddr>& feeAddrArray, const std::vector<TriggerConfig>& triggerConfigArray);

    __host__ __device__ bool getKey(uint8_t ip3rd, uint8_t ip4th, uint8_t ch, uint32_t& key) const;

    __host__ __device__ TriggerConfig getTriggerConfig(uint32_t key) const;

    uint32_t getSizeSpaceOfKey() const{ return sizeSpaceOfKey; }

    const TriggerConfig* getTriggerConfigArray() const{ return triggerConfigArray; }

    void setTriggerConfigArray(TriggerConfig* ptr){ triggerConfigArray = ptr; }

private:
    uint8_t minIP3rd{};
    uint8_t maxIP3rd{};
    uint8_t minIP4th{};
    uint8_t maxIP4th{};
    uint8_t minCh{};
    uint8_t maxCh{};

    uint16_t sizeSpaceOfIP3rd{};
    uint16_t sizeSpaceOfIP4th{};
    uint16_t sizeSpaceOfCh{};

    uint32_t sizeSpaceOfKey{};

    TriggerConfig* triggerConfigArray{nullptr};
};

} // namespace cudamap

#endif