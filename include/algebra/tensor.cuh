
#ifndef TENSOR
#define TENSOR

#include <stdexcept>
#include <vector>
#include "types.hpp"

namespace IdioticML{


struct Range {
    unsigned int start; // inlcusive
    unsigned int end;   // inclusive
};

class Tensor {
private:
    // Allocate space for tensor, where it will be allocated depends on member is_onCUDA.
    // If the src_ptr is provided, the memory is also copy to the tensor.
    void allocate(const void *src_ptr = nullptr);

protected:
    std::vector<unsigned int> dimensions;

    // This can be on either HOST or DEVICE (GPU) depending on the parameters passed to constructor
    void *tensor; 
    unsigned int totalEntries;
    TensorType type;
    bool is_onCUDA = false;

    // The number of threads per block when using CUDA
    static const unsigned int threadsPerBlock = 512;

public:
    
    // Virtual destructor for polymorphism
    virtual ~Tensor();
    Tensor() = default;

    /**
     * Init a tensor using dimensions.
     * 
     * If the src_tensor is provided, 
     */
    Tensor (const std::vector<unsigned int>& dimensions, 
            TensorType type,
            const void *src_tensor = nullptr,
            bool to_cuda = false);

    void printTensor();

    // When performing tensor's arithmetic operations, both tensors must be on the same device,
    // both on GPU or both on CPU.

    // Overload operator +
    Tensor operator+(const Tensor& other) const;
};

}

#endif
