#ifndef CPUHANDLER
#define CPUHANDLER

#include "device_adapter.hpp"

namespace IdioticML{

class CPU_adapter : public DeviceAdapter{
public:
    CPU_adapter(const Device& device);
    void allocate(void **ptr, size_t num_bytes, const void *src = nullptr) override;
    void deallocate(void **ptr) override;
    void copyTo(void *dest, const void *src, size_t num_bytes) override;

    bool isGPU() override;
    bool isCPU() override;
    int getGPU_id() override;
};

}

#endif