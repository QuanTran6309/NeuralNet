
#ifndef CUDA_WRAP
#define CUDA_WRAP


#include <vector>

/**
 * Specify the number of threads per block to be allocated for a kernel
 * 
 * This is primarily used for tensor's opeartions.
 * 
 * Because Tensor class is too generic, it is quite difficult to come up with an
 * optimal configuration to launch the CUDA kernel so I make it a concrete number.
 */
#define THREADSPERBLOCK 512

namespace IdioticML {
namespace CUDA {

/**
 * Check if there exists an NVIDIA GPU or not
 */
bool is_GPUreadyToUse();


/**
 * A wraper around CUDA kernel.
 * 
 * Each wrap will basically do the following:
 *      - Allocate memory on GPU
 *      - Copy source pointer to corresponding memory on GPU
 *      - Launch the corresponding kernel to compute
 *      - Copy the computed result back to a given pointer in CPU
 *      - Free memory in GPU
 * 
 * NOTE: any pointer having syntax "h_something" hints that it is on CPU
 * WARNING: all static methods in this wrap does not care and check any restriction regarding any operations they do.
 *          For example, the matrix multiplication wrap does not check if the given dimension of matrix1 and matrix2
 *          are valid but just asume and execute.
 */
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

    
    /**
     * Perform matrix multiplication using GPU
     * 
     * This is just a wrap and it does not check if two given matrices have valid dimensions to 
     * perform multiplication or not.
     */
    static void matrixMultiplication(const T *h_matrix1,
                                     const std::vector<unsigned int>& h_matrix1_Dim,
                                     const T *h_matrix2,
                                     const std::vector<unsigned int>& h_matrix2_Dim,
                                     T *h_matrix);
};

}
}


#endif
