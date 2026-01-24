#ifndef GPUBACKEND
#define GPUBACKEND

#include "OpsBackend.hpp"
#include "adapter/device.hpp"
#include <cublas_v2.h>


namespace IdioticML{

class GpuBackend : public OpsBackend {
private:
    cublasHandle_t handler;

public:
    GpuBackend(const Device& device);
    ~GpuBackend();
    
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