
#ifndef CUDA_WRAP
#define CUDA_WRAP

#include "types.hpp"

namespace IdioticML {


class CUDA {
private:
    // The number of threads per block when using CUDA
    static const unsigned int threadsPerBlock = 512;

public:

    /**
     * Perform tensor addition 
     */
    static void tensorAddition(const void *h_tensor1, 
                               const void *h_tensor2, 
                               void *h_tensor, 
                               TensorType type,
                               unsigned int totalEntries);
    
};

}


#endif
