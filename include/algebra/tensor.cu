
#include <cuda_runtime.h>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <iostream>

#include "tensor.cuh"

namespace IdioticML {


// This is a generic tensor addition kernel so we have to consider the tensor as a 1D array
template<typename T>
__global__
void tensorAddKernel(const T *tensor1, 
                     const T *tensor2, 
                     T *tensor, 
                     unsigned int totalEntries)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N){
        tensor[i] = tensor1[i] + tensor2[i];
    }
}


Tensor::~Tensor(){
    if (is_onCUDA){
        cudaFree(this->tensor);
    }
    else{
        free(this->tensor);
    }
}


void Tensor::allocate(const void *src_ptr){
    // The number of bytes of the tensor, not just the number of entries 
    size_t size = TensorTypeSize.at(this->type) * this->totalEntries;

    // Allocate and Assign value for tensor
    if (this->is_onCUDA){
        // for GPU
        cudaMalloc(&this->tensor, size);
        if (src_ptr != nullptr){
            cudaMemcpy(this->tensor, src_ptr, size, cudaMemcpyHostToDevice);
        }
    }
    else {
        // for CPU
        this->tensor = this->tensor = malloc(size);
        if (src_ptr != nullptr){
            memcpy(this->tensor, src_ptr, size);
        }
    }
}


Tensor::Tensor (const std::vector<unsigned int>& dimensions, 
                TensorType type, 
                const void *src_tensor,
                bool to_cuda)
{
    if (dimensions.size() == 0){
        throw std::runtime_error("The dimensions cannot be empty");
    }

    this->dimensions = dimensions;
    this->totalEntries = std::accumulate(this->dimensions.begin(), this->dimensions.end(), 1, std::multiplies<unsigned int>());
    this->type = type;
    this->is_onCUDA = to_cuda;

    if (this->totalEntries == 0){
        throw std::runtime_error("Dimension of size 0 is not allowed");
    }

    // Allocate and copy data to tensor
    this->allocate(src_tensor);
}


void Tensor::printTensor(){
    size_t size = TensorTypeSize.at(this->type) * this->totalEntries;
    void *buffer = malloc(size);
    
    cudaMemcpy(buffer, this->tensor, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < this->totalEntries; i++){
        std::cout << ((int *)buffer)[i] << " ";
    }
    std::cout << std::endl;
}


Tensor Tensor::operator+(const Tensor& other) const 
{
    if (this->is_onCUDA ^ other.is_onCUDA){
        throw std::runtime_error("Both tensors must be on either CPU or GPU");
    }

    // Create new tensor instance
    Tensor newTensor(this->dimensions, this->type, nullptr, this->is_onCUDA);
    
    // The number of threads per block is set to a static number
    // So just need to calculate the number of blocks.
    int numBlocks = this->totalEntries / Tensor::threadsPerBlock + 1;

    // Launch the kernel
    switch (this->type)
    {
    case TensorType::INT:
        tensorAddKernel<int><<< numBlocks, Tensor::threadsPerBlock >>>((int *)this->tensor, 
                                                                       (int *)other.tensor, 
                                                                       (int *)newTensor.tensor, 
                                                                       this->totalEntries);
        break;
    case TensorType::FLOAT:
        tensorAddKernel<float><<< numBlocks, Tensor::threadsPerBlock >>>((float *)this->tensor, 
                                                                         (float *)other.tensor, 
                                                                         (float *)newTensor.tensor, 
                                                                         this->totalEntries);
        break;
    case TensorType::INT:
        tensorAddKernel<int><<< numBlocks, Tensor::threadsPerBlock >>>((int *)this->tensor, 
                                                                       (int *)other.tensor, 
                                                                       (int *)newTensor.tensor, 
                                                                       this->totalEntries);
        break;
    default:
        throw std::runtime_error("Addition operation does not support this data type");
    }
    
    cudaDeviceSynchronize();
    
    return newTensor;
}

}


