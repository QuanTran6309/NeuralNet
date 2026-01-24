#ifndef CPUBACKEND
#define CPUBACKEND

#include "OpsBackend.hpp"
#include "adapter/device.hpp"

namespace IdioticML{

class CpuBackend : public OpsBackend {
public:
    CpuBackend(const Device& device);
    
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