
#ifndef CUDA_WRAP
#define CUDA_WRAP

#include <cuda_runtime.h>
#include "types.hpp"


namespace IdioticML {

// Generic kernel for adding two tensors
template<typename T>
__global__
void tensorAdditionKernel(const T *d_tensor1, const T *d_tensor2, T *d_tensor, unsigned int totalEntries){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < totalEntries){
        d_tensor[i] = d_tensor1[i] + d_tensor2[i];
    }
}

class CUDA {
private:
    // The number of threads per block when using CUDA
    static const unsigned int threadsPerBlock = 512;

public:

    static void tensorAddition(const void *h_tensor1, 
                               const void *h_tensor2, 
                               void *h_tensor, 
                               TensorType type,
                               unsigned int totalEntries);
    
};

}


#endif
