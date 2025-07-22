#include "memory/cpu_adapter.hpp"
#include "utils/logger.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

namespace IdioticML{

CPU_adapter::CPU_adapter(const Device& device) : DeviceAdapter(device){
    if (device.type != DeviceType::CPU){
        LOGEXCEPTION("CPU Adapter requires device of type CPU")
    }
}

void CPU_adapter::allocate(void **ptr, size_t num_bytes, const void *src) {
    *ptr = std::malloc(num_bytes);
    if (*ptr == nullptr) {
        LOGEXCEPTION("Fail to allocate memory on RAM.")
    }

    if (src == nullptr){
        return;
    }

    cudaPointerAttributes attr;
    CUDA_ERR_CHECK(cudaPointerGetAttributes(&attr, src));

    // If *src is on the GPU, then must use CUDA method
    if (attr.type == cudaMemoryTypeDevice){
        CUDA_ERR_CHECK(cudaMemcpy(*ptr, src, num_bytes, cudaMemcpyDeviceToHost));
    }
    else {
        if (attr.type == cudaMemoryTypeManaged){
            // If the src pointer is on the unified memory, 
            // we need to make sure any async operations about that chunk of memory
            // is done before reading.
            cudaDeviceSynchronize();
        }
        std::memcpy(*ptr, src, num_bytes);
    }
}

void CPU_adapter::deallocate(void **ptr) {
    std::free(*ptr);
    *ptr = nullptr;
}


void CPU_adapter::copyTo(void *dest, const void *src, size_t num_bytes) {
    cudaPointerAttributes attr;
    CUDA_ERR_CHECK(cudaPointerGetAttributes(&attr, dest));

    // If *src is on the GPU, then must use CUDA method
    if (attr.type == cudaMemoryTypeDevice){
        CUDA_ERR_CHECK(cudaMemcpy(dest, src, num_bytes, cudaMemcpyDeviceToHost));
    }
    else {
        if (attr.type == cudaMemoryTypeManaged){
            // If the src pointer is on the unified memory, 
            // we need to make sure any async operations about that chunk of memory
            // is done before reading.
            cudaDeviceSynchronize();
        }
        std::memcpy(dest, src, num_bytes);
    }
}


bool CPU_adapter::isGPU() {return false;}
bool CPU_adapter::isCPU() {return true;}
int CPU_adapter::getGPU_id(){
    LOGEXCEPTION("The current device is CPU, cannot get GPU's ID")
}




}