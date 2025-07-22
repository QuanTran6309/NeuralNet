#include "memory/gpu_adapter.hpp"
#include "utils/logger.cuh"
#include <iostream>
#include <cuda_runtime.h>


namespace IdioticML{

GPU_adapter::GPU_adapter(const Device& device) : DeviceAdapter(device) {
    if (device.type != DeviceType::GPU){
        LOGEXCEPTION("GPU Adapter requires device of type GPU")
    }
}

void GPU_adapter::allocate(void **ptr, size_t num_bytes, const void *src) {
    CUDA_ERR_CHECK(cudaMalloc(ptr, num_bytes));
    if (src == nullptr){
        return;
    }

    cudaSetDevice(this->device.id);

    cudaPointerAttributes attr;
    CUDA_ERR_CHECK(cudaPointerGetAttributes(&attr, src));
    cudaMemcpyKind cpyKind = cudaMemcpyHostToDevice;

    // If *src is on the GPU, then must use CUDA method
    if (attr.type == cudaMemoryTypeDevice){
        cpyKind = cudaMemcpyDeviceToDevice;
    }
    else if (attr.type == cudaMemoryTypeManaged){
        // If the src pointer is on the unified memory, 
        // we need to make sure any async operations about that chunk of memory
        // is done before reading.
        cudaDeviceSynchronize();
    }

    CUDA_ERR_CHECK(cudaMemcpy(*ptr, src, num_bytes, cpyKind));
}

void GPU_adapter::deallocate(void **ptr) {
    CUDA_ERR_CHECK(cudaFree(*ptr));
    *ptr = nullptr;
}

void GPU_adapter::copyTo(void *dest, const void *src, size_t num_bytes){
    cudaPointerAttributes attr;
    CUDA_ERR_CHECK(cudaPointerGetAttributes(&attr, dest));

    // If *src is on the GPU, then must use CUDA method
    if (attr.type == cudaMemoryTypeDevice){
        CUDA_ERR_CHECK(cudaMemcpy(dest, src, num_bytes, cudaMemcpyDeviceToDevice));
    }
    else {
        if (attr.type == cudaMemoryTypeManaged){
            // If the src pointer is on the unified memory, 
            // we need to make sure any async operations about that chunk of memory
            // is done before reading.
            cudaDeviceSynchronize();
        }
        CUDA_ERR_CHECK(cudaMemcpy(dest, src, num_bytes, cudaMemcpyDeviceToHost));
    }
}



bool GPU_adapter::isGPU() {return true;}
bool GPU_adapter::isCPU() {return false;}
int GPU_adapter::getGPU_id(){
    return this->device.id;
}

}