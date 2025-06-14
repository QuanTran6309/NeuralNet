#include "cuda.cuh"
#include <stdexcept>

namespace IdioticML {




void CUDA::tensorAddition(const void *h_tensor1, 
                          const void *h_tensor2, 
                          void *h_tensor, 
                          TensorType type,
                          unsigned int totalEntries)
{
    // The number of threads per block is set to a static number
    // So just need to calculate the number of blocks.
    unsigned int numBlocks = totalEntries / CUDA::threadsPerBlock + 1;

    // Allocate and copy data to GPU
    void *d_tensor1;
    void *d_tensor2;
    void *d_tensor;

    size_t size = totalEntries * TensorTypeSize.at(type);
    cudaMalloc(&d_tensor1, size);
    cudaMalloc(&d_tensor2, size);
    cudaMalloc(&d_tensor, size);
    cudaMemcpy(d_tensor1, h_tensor1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tensor2, h_tensor2, size, cudaMemcpyHostToDevice);

    // Launch the kernel
    switch (type)
    {
    case TensorType::INT:
        tensorAdditionKernel<int><<<numBlocks, CUDA::threadsPerBlock>>>((int *)d_tensor1, 
                                                                        (int *)d_tensor2, 
                                                                        (int *)d_tensor, 
                                                                        totalEntries);
        break;
    case TensorType::FLOAT:
        tensorAdditionKernel<float><<<numBlocks, CUDA::threadsPerBlock>>>((float *)d_tensor1, 
                                                                          (float *)d_tensor2, 
                                                                          (float *)d_tensor, 
                                                                          totalEntries);
        break;
    default:
        throw std::runtime_error("Encounter unsupported datatype");
    }
    cudaDeviceSynchronize();
    
    // Copy the result 
    cudaMemcpy(h_tensor, d_tensor, size, cudaMemcpyDeviceToHost);

    // Free the pointer in GPU
    cudaFree(d_tensor1);
    cudaFree(d_tensor2);
    cudaFree(d_tensor);
}




}