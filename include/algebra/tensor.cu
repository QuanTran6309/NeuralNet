
#include <cuda_runtime.h>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <iostream>

#include "tensor.cuh"

std::unordered_map<TensorType, unsigned int> MemSizeMap = {
    {FLOAT, 4},
    {DOUBLE, 8},
    {INT, 4},
    {UINT, 4}
};


__global__ void vectorAddKernel(const int* A, const int* B, int* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
/*
void vectorAdd(const float* A, const float* B, float* C, int N) {
    float *d_A, *d_B, *d_C;
    size_t size = N * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
    */


Tensor::~Tensor(){
    if (is_onCUDA){
        cudaFree(this->tensor);
    }
    else{
        free(this->tensor);
    }
}

Tensor::Tensor(const std::vector<unsigned int>& dimensions, 
               TensorType type,
                bool to_cuda)
{
    if (!(MemSizeMap.find(type) != MemSizeMap.end())){
        throw std::runtime_error("Unsupported type");
    }
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

    size_t size = MemSizeMap[this->type] * this->totalEntries;
    // Allocate and Assign value for tensor
    if (this->is_onCUDA){
        cudaMalloc(&this->tensor, size);
    }
    else {
        this->tensor = this->tensor = malloc(MemSizeMap[this->type] * this->totalEntries);
    }
}


Tensor::Tensor (const void *src_tensor, 
                const std::vector<unsigned int>& dimensions, 
                TensorType type, 
                bool to_cuda)
{
    if (!(MemSizeMap.find(type) != MemSizeMap.end())){
        throw std::runtime_error("Unsupported type");
    }
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

    size_t size = MemSizeMap[this->type] * this->totalEntries;

    // Allocate and Assign value for tensor
    if (this->is_onCUDA){
        cudaMalloc(&this->tensor, size);
        cudaMemcpy(this->tensor, src_tensor, size, cudaMemcpyHostToDevice);
    }
    else {
        this->tensor = this->tensor = malloc(size);
        memcpy(this->tensor, src_tensor, size);
    }
}


void Tensor::printTensor(){
    size_t size = MemSizeMap[this->type] * this->totalEntries;
    void *buffer = malloc(size);
    
    cudaMemcpy(buffer, this->tensor, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < this->totalEntries; i++){
        std::cout << ((int *)buffer)[i] << " ";
    }
    std::cout << std::endl;
}


Tensor Tensor::operator+(const Tensor& other) const {
    // Create new tensor instance
    Tensor newTensor(this->dimensions, this->type, this->is_onCUDA);
    
    vectorAddKernel<<<1, this->totalEntries>>>((int *)this->tensor, (int *)other.tensor, (int *)newTensor.tensor, (int)this->totalEntries);
    cudaDeviceSynchronize();
    
    return newTensor;
}



