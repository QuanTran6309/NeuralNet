#include "adapter/memory/cpu_memory.hpp"
#include "utils/logger.cuh"
#include <cuda_runtime.h>
#include <cstring>
#include <utility>

namespace IdioticML{

CpuMemory::CpuMemory(const Device& device) : DeviceMemory(device){
    if (device.type != DeviceType::CPU){
        LOGEXCEPTION("CPU Adapter requires device of type CPU")
    }
}

void CpuMemory::allocate(void **ptr, size_t num_bytes, const void *src) {
    *ptr = std::malloc(num_bytes);
    if (*ptr == nullptr) {
        LOGEXCEPTION("Fail to allocate memory on RAM.")
    }
    std::memset(*ptr, 0, num_bytes); // Very important

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

void CpuMemory::deallocate(void **ptr) {
    std::free(*ptr);
    *ptr = nullptr;
}


void CpuMemory::copyTo(void *dest, const void *src, size_t num_bytes) {
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




template<typename T, std::size_t... Idx>
inline void simultAssign(T *dest, 
                         const T *src, 
                         const unsigned int i,
                         const unsigned int *indices,
                         std::index_sequence<Idx...>)                         
{
    ((dest[i + Idx] = src[indices[i + Idx]]), ...);
}
void CpuMemory::copyAtIndices(void *dest, 
                                const void *src, 
                                const unsigned int *indices, 
                                unsigned int numberOfIndices,
                                const TensorType& type)
{
    unsigned int remain = numberOfIndices % CHUNK;
    unsigned int iterations = numberOfIndices / CHUNK;


    auto copyFunctor = [&](auto type_t){
        using T = decltype(type_t);

        T *destPtr = static_cast<T *>(dest);
        const T *srcPtr = static_cast<const T *>(src);

        for (unsigned int i = 0; i < iterations; i++){
            simultAssign<T>(destPtr, srcPtr, i, indices, std::make_index_sequence<CHUNK>{});
        }
        
        // Handle the remaining.
        for (unsigned int i = iterations * CHUNK; i < numberOfIndices; i++){
            destPtr[i] = srcPtr[indices[i]];
        }
    };

    type_dispatcher(type, copyFunctor);
}




bool CpuMemory::isGPU() {return false;}
bool CpuMemory::isCPU() {return true;}
int CpuMemory::getGPU_id(){
    LOGEXCEPTION("The current device is CPU, cannot get GPU's ID")
}


void CpuMemory::add(void *dest, 
                     const void *src1, 
                     const void *src2, 
                     unsigned int numberOfEntries, 
                     const TensorType& type) 
{

    auto add_functor = [&](auto type_t) {
        using T = decltype(type_t);

        std::cout << "Tensor addition" << std::endl;
        T *result = static_cast<T *>(dest);
        const T *src_1 = static_cast<const T *>(src1);
        const T *src_2 = static_cast<const T *>(src2);

        for (unsigned int i = 0; i < numberOfEntries; i++){
            result[i] = src_1[i] + src_2[i];
        }
    };

    type_dispatcher(type, add_functor);
}   


void CpuMemory::mult(int m, int n, int k,
                       const void *src1,
                       const void *src2,
                       void *dest,
                       const TensorType& type)
{

    auto mult_functor = [&](auto type_t){
        using T = decltype(type_t);

        T *result = static_cast<T *>(dest);
        const T *src_1 = static_cast<const T *>(src1);
        const T *src_2 = static_cast<const T *>(src2);

        for (unsigned int row = 0; row < n; row++){
            for (unsigned int col = 0; col < m; col++){
                for (unsigned int itr = 0; itr < k; itr++){
                    result[col + row * m] += ( src_1[row * k + itr] * src_2[col + itr * m] );
                }
            }
        }
    };

    type_dispatcher(type, mult_functor);
}


}