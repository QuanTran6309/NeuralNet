#include "cuda.cuh"
#include <stdexcept>
#include <cuda_runtime.h>

namespace IdioticML {




/**
 * Check if there exists an NVIDIA GPU or not
 */
bool CUDA::isGPU_available()
{
    int gpuCount = 0;

    cudaError_t err = cudaGetDeviceCount(&gpuCount);
    if (err != cudaSuccess){
        throw std::runtime_error(cudaGetErrorString(err));
    }

    if (gpuCount > 0){
        return true;
    }
    return false;
}



// Generic kernel for adding two tensors
template<typename T>
__global__
void tensorAdditionKernel(const T *d_tensor1, 
                          const T *d_tensor2, 
                          T *d_tensor, 
                          unsigned int totalEntries)
{
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < totalEntries){
        d_tensor[i] = d_tensor1[i] + d_tensor2[i];
    }
}
template<typename T>
void CUDA::KernelWrap<T>::tensorAddition(const T *h_tensor1, 
                                         const T *h_tensor2, 
                                         T *h_tensor, 
                                         unsigned int totalEntries)
{
    // The number of threads per block is set to a static number
    // So just need to calculate the number of blocks.
    unsigned int numBlocks = totalEntries / CUDA::threadsPerBlock + 1;

    // Allocate and copy data to GPU
    T *d_tensor1;
    T *d_tensor2;
    T *d_tensor;

    size_t size = totalEntries * sizeof(T);
    cudaMalloc(&d_tensor1, size);
    cudaMalloc(&d_tensor2, size);
    cudaMalloc(&d_tensor, size);
    cudaMemcpy(d_tensor1, h_tensor1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tensor2, h_tensor2, size, cudaMemcpyHostToDevice);

    // Launch the kernel
    tensorAdditionKernel<T><<<numBlocks, CUDA::threadsPerBlock>>>(d_tensor1, 
                                                                  d_tensor2, 
                                                                  d_tensor, 
                                                                  totalEntries);
    cudaDeviceSynchronize();
    
    // Copy the result 
    cudaMemcpy(h_tensor, d_tensor, size, cudaMemcpyDeviceToHost);

    // Free the pointer in GPU
    cudaFree(d_tensor1);
    cudaFree(d_tensor2);
    cudaFree(d_tensor);
}



// Generic kernel for subtracting two tensors
template<typename T>
__global__
void tensorSubtractionKernel(const T *d_tensor1, 
                             const T *d_tensor2, 
                             T *d_tensor, 
                             unsigned int totalEntries)
{
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < totalEntries){
        d_tensor[i] = d_tensor1[i] - d_tensor2[i];
    }
}
template<typename T>
void CUDA::KernelWrap<T>::tensorSubtraction(const T *h_tensor1, 
                                            const T *h_tensor2, 
                                            T *h_tensor, 
                                            unsigned int totalEntries)
{
    // The number of threads per block is set to a static number
    // So just need to calculate the number of blocks.
    unsigned int numBlocks = totalEntries / CUDA::threadsPerBlock + 1;

    // Allocate and copy data to GPU
    T *d_tensor1;
    T *d_tensor2;
    T *d_tensor;

    size_t size = totalEntries * sizeof(T);
    cudaMalloc(&d_tensor1, size);
    cudaMalloc(&d_tensor2, size);
    cudaMalloc(&d_tensor, size);
    cudaMemcpy(d_tensor1, h_tensor1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tensor2, h_tensor2, size, cudaMemcpyHostToDevice);

    // Launch the kernel
    tensorSubtractionKernel<T><<<numBlocks, CUDA::threadsPerBlock>>>(d_tensor1, 
                                                                     d_tensor2, 
                                                                     d_tensor, 
                                                                     totalEntries);
    cudaDeviceSynchronize();
    
    // Copy the result 
    cudaMemcpy(h_tensor, d_tensor, size, cudaMemcpyDeviceToHost);

    // Free the pointer in GPU
    cudaFree(d_tensor1);
    cudaFree(d_tensor2);
    cudaFree(d_tensor);
}






// I find this god damn ridiculous
template class CUDA::KernelWrap<int>;
template class CUDA::KernelWrap<float>;
template class CUDA::KernelWrap<double>;


}