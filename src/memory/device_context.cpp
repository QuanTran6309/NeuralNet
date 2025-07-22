#include "memory/device_context.hpp"
#include "memory/cpu_adapter.hpp"
#include "memory/gpu_adapter.hpp"
#include <cstring>
#include <iostream>
namespace IdioticML {


DeviceContext::DeviceContext(const Device& device){
    if (device.type == DeviceType::GPU){
        this->adapter = std::make_shared<GPU_adapter>(device);
    }
    else {
        this->adapter = std::make_shared<CPU_adapter>(device);
    }
}

void DeviceContext::allocate(void *ptr, size_t num_bytes, const void *src){
    this->adapter->allocate((void**)ptr, num_bytes, src);
}
void DeviceContext::deallocate(void *ptr){
    this->adapter->deallocate((void**)ptr);
}

// Migrate data from GPU to CPU
void DeviceContext::changeContextToCPU(void **ptr, size_t num_bytes){
    if (this->adapter->isCPU()){
        return;
    }

    Device device(DeviceType::CPU);
    void *buffer;
    
    // Temporarily migrate the data to buffer
    std::shared_ptr<DeviceAdapter> cpuAdapter = std::make_shared<CPU_adapter>(device);
    cpuAdapter->allocate(&buffer, num_bytes, *ptr);
    
    // Free the current ptr
    // This method also nullify the ptr
    this->adapter->deallocate(ptr);

    // Re-assign the ptr to the buffer & change the adapter to a new one
    *ptr = buffer;
    this->adapter = cpuAdapter;
}

// Migrate from CPU to GPU, or from one GPU to another
void DeviceContext::changeContextToGPU(void **ptr, 
                                       size_t num_bytes, 
                                       int newGPU_id){
    
    if (this->adapter->isGPU() && this->adapter->getGPU_id() == newGPU_id){
        return;
    }

    Device device(DeviceType::GPU, newGPU_id);
    void *buffer;
    
    // Temporarily migrate the data to buffer
    std::shared_ptr<DeviceAdapter> gpuAdapter = std::make_shared<GPU_adapter>(device);
    gpuAdapter->allocate(&buffer, num_bytes, *ptr);
    
    // Free the current ptr
    // This method also nullify the ptr
    // Must deallocate before assigning a new adapter because it may call different free method on a pointer on different device.
    this->adapter->deallocate(ptr);

    // Re-assign the ptr to the buffer & change the adapter to a new one
    *ptr = buffer;
    this->adapter = gpuAdapter;
}



void DeviceContext::copyTo(void *dest, const void* src, size_t num_bytes){
    this->adapter->copyTo(dest, src, num_bytes);
}


DeviceType DeviceContext::getDeviceType(){
    return this->adapter->getDeviceType();
}



}