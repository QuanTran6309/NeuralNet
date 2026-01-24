
#ifndef GPUMEMORY
#define GPUMEMORY

#include "device_memory.hpp"
#include <cublas_v2.h>

namespace IdioticML{

class GpuMemory : public DeviceMemory{
    cublasHandle_t handler;
    
public:
    GpuMemory(const Device& device);
    ~GpuMemory();

    void allocate(void **ptr, size_t num_bytes, const void *src = nullptr) override;
    void deallocate(void **ptr) override;
    void copyTo(void *dest, const void *src, size_t num_bytes) override;
    void copyAtIndices(void *dest, 
                       const void *src, 
                       const unsigned int *indices, 
                       unsigned int numberOfIndices, 
                       const TensorType& type) override;

    bool isGPU() override;
    bool isCPU() override;
    int getGPU_id() override;


    // Arithmetic
    void add(void *dest, 
             const void *src1, 
             const void *src2, 
             unsigned int numberOfEntries, 
             const TensorType& type) override;
             
    void mult(int m, int n, int k,
              const void *src1,
              const void *src2,
              void *dest,
              const TensorType& type) override;


};

}

#endif
