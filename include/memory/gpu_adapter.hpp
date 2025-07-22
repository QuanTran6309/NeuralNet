
#ifndef GPUHANDLER
#define GPUHANDLER

#include "device_adapter.hpp"

namespace IdioticML{

class GPU_adapter : public DeviceAdapter{
public:
    GPU_adapter(const Device& device);
    void allocate(void **ptr, size_t num_bytes, const void *src = nullptr) override;
    void deallocate(void **ptr) override;
    void copyTo(void *dest, const void *src, size_t num_bytes) override;

    bool isGPU() override;
    bool isCPU() override;
    int getGPU_id() override;
};

}

#endif
