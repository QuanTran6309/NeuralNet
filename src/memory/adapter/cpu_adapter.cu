#include "memory/adapter/cpu_adapter.hpp"
#include "utils/logger.cuh"
#include <cuda_runtime.h>
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



template<typename T>
inline void additionHelper(T *dest, const T *src1, const T *src2, unsigned int numberOfEntries){
    for (unsigned int i = 0; i < numberOfEntries; i++){
        dest[i] = src1[i] + src2[i];
    }
}
void CPU_adapter::add(void *dest, 
                     const void *src1, 
                     const void *src2, 
                     unsigned int numberOfEntries, 
                     const TensorType& type) 
{
    // This switch case is ridiculous but I have no better way.
    switch (type)
    {
    case TensorType::FLOAT: {
        additionHelper<float>((float *)dest, (float *)src1, (float *)src2, numberOfEntries);
        break;
    }
    case TensorType::DOUBLE: {
        additionHelper<double>((double *)dest, (double *)src1, (double *)src2, numberOfEntries);
        break;
    }
    default:
        LOGEXCEPTION("Unknown datatype");
    }
}   


template<typename T>
inline void multiplicationHelper(int m, int n, int k,
                                 const T *src1,
                                 const T *src2,
                                 T *dest)
{
    for (unsigned int row = 0; row < n; row++){
        for (unsigned int col = 0; col < m; col++){
            for (unsigned int itr = 0; itr < k; itr++){
                dest[col + row * m] += ( src1[row * n + itr] * src2[col + itr * m] );
            }
        }
    }
}
void CPU_adapter::mult(int m, int n, int k,
                       const void *src1,
                       const void *src2,
                       void *dest,
                       const TensorType& type)
{
    switch (type)
    {
    case TensorType::FLOAT:{
        multiplicationHelper<float>(m, n, k, (float *)src1, (float *)src2, (float *)dest);
        break;
    }
    case TensorType::DOUBLE:{
        multiplicationHelper<double>(m, n, k, (double *)src1, (double *)src2, (double *)dest);
        break;
    }
    default:
        LOGEXCEPTION("Unknown datatype")
    }
}


}