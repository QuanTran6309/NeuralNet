#ifndef DEVICEHANDLER
#define DEVICEHANDLER

#include "device.hpp"

namespace IdioticML{
/**
 * In charge of handling operations relating to a specific device, CPU or GPU
 */
class DeviceAdapter {
protected:
    Device device;
public:
    DeviceAdapter(const Device& device);
    virtual ~DeviceAdapter() = default;
    virtual void allocate(void **ptr, size_t num_bytes, const void *src = nullptr) = 0;
    virtual void deallocate(void **ptr) = 0;

    /**
     * Copy from src to dest
     * 
     * @param dest: can be either on the GPU or CPU, the method will detect this.
     * @param src: must be on the same device with the device passed into the constructor.
     */
    virtual void copyTo(void *dest, const void *src, size_t num_bytes) = 0;

    virtual bool isGPU() = 0;
    virtual bool isCPU() = 0;
    virtual int getGPU_id() = 0;

    DeviceType getDeviceType();
};

}

#endif