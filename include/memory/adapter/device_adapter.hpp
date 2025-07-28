#ifndef DEVICEHANDLER
#define DEVICEHANDLER

#include "memory/device.hpp"
#include <vector>

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

    DeviceType getDeviceType();


    /**
     * Allocate memory for ptr and copy from src to ptr.
     * src pointer can be either on CPU or GPU does not matter.
     */
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

    // Arithmetic operations
    // None of these operations will perform any checks before doing the operations
    virtual void add(void *dest, 
                     const void *src1, 
                     const void *src2, 
                     unsigned int numberOfEntries, 
                     const TensorType& type) = 0;

    virtual void mult(int m, int n, int k,
                      const void *src1,
                      const void *src2,
                      void *dest,
                      const TensorType& type) = 0;

};

}

#endif