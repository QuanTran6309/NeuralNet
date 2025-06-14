

#include <algorithm>
#include <numeric>
#include <iostream>
#include <stdexcept>

#include "tensor.cuh"
#include "cuda/cuda.cuh"


namespace IdioticML {


Tensor::~Tensor(){
    free(this->tensor);
}


void Tensor::allocate(const void *src_ptr){
    // The number of bytes of the tensor, not just the number of entries 
    size_t size = TensorTypeSize.at(this->type) * this->totalEntries;
    this->tensor = malloc(size);

    // If the src_ptr is provided, copy value to this tensor.
    if (src_ptr != nullptr){
        memcpy(this->tensor, src_ptr, size);
    }
}


Tensor::Tensor (const std::vector<unsigned int>& dimensions, 
                TensorType type, 
                const void *src_tensor)
{
    if (dimensions.size() == 0){
        throw std::runtime_error("The dimensions cannot be empty");
    }

    this->dimensions = dimensions;
    this->totalEntries = std::accumulate(this->dimensions.begin(), this->dimensions.end(), 1, std::multiplies<unsigned int>());
    this->type = type;

    if (this->totalEntries == 0){
        throw std::runtime_error("Dimension of size 0 is not allowed");
    }

    // Allocate and copy data to tensor
    this->allocate(src_tensor);
}


void Tensor::printTensor(){
    for (int i = 0; i < this->totalEntries; i++){
        std::cout << ((int *)this->tensor)[i] << " ";
    }
    std::cout << std::endl;
}


Tensor Tensor::operator+(const Tensor& other) const 
{

    // Create new tensor instance
    Tensor newTensor(this->dimensions, this->type, nullptr);
    
    // Launch the kernel
    CUDA::tensorAddition(this->tensor, other.tensor, newTensor.tensor, this->type, this->totalEntries);

    return newTensor;
}

}


