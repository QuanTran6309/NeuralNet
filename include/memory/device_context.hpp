
#ifndef DEVICEMANAGER
#define DEVICEMANAGER
#include <memory>
#include "device_adapter.hpp"

namespace IdioticML{


struct DeviceContext {
private:
    std::shared_ptr<DeviceAdapter> adapter;

public:
    DeviceContext() = default;
    DeviceContext(const Device& device);

    /**
     * Tensor's data migration methods,
     * 
     * Call ones of these two methods to migrate data from CPU to GPU, GPU to CPU, or accross different GPU
     * if you have multiple GPU.
     */
    void changeContextToGPU(void **ptr, size_t num_bytes, int newGPU_id = 0);
    void changeContextToCPU(void **ptr, size_t num_bytes);


    // From this line, all method below are just wrapper around the methods with the same name of DeviceAdapter

    /**
     * Allocate memory for the given pointer. 
     * @param ptr: must pass address of the pointer but not the pointer itself here.
     * @param num_bytes: the number of bytes to be allocated
     * @param src: the source data to be copy from to ptr. Can be left nullptr, the method wont copy anything.
     * 
     */
    void allocate(void *ptr, size_t num_bytes, const void *src = nullptr);

    /**
     * Deallocate memory for the given pointer. 
     * @param ptr: must pass address of the pointer but not the pointer itself here. This pointer is assigned to nullptr after being freed.
     * 
     */
    void deallocate(void *ptr);
    
    /**
     * Copy data from the src to dest.
     * 
     * @param dest: can be on either CPU or GPU. The method will detect its device.
     * @param src:  must be on the same device with the device passed into the constructor.
     */
    void copyTo(void *dest, const void* src, size_t num_bytes);
    DeviceType getDeviceType();

};



}


#endif