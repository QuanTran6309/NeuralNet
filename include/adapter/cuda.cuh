
#ifndef CUDA_WRAP
#define CUDA_WRAP

#include "types.hpp"

namespace IdioticML {
namespace CUDA {

/**
 * Specify the number of threads per block to be allocated for a kernel
 */
const unsigned int threadsPerBlock = 512;

/**
 * Check if there exists an NVIDIA GPU or not
 */
bool isGPU_available();


template<typename T>
class KernelWrap{
public:
    /**
     * Perform tensor addition using GPU
     * 
     * Technically it is a wrap for the kernel
     */
    static void tensorAddition(const T *h_tensor1, 
                               const T *h_tensor2, 
                               T *h_tensor, 
                               unsigned int totalEntries);

    /**
     * Perform tensor subtraction using GPU
     * 
     * Technically it is a wrap for a kernel
     */
    static void tensorSubtraction(const T *h_tensor1, 
                                  const T *h_tensor2, 
                                  T *h_tensor, 
                                  unsigned int totalEntries);
};

}
}


#endif
