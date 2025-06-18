#include "cuda.cuh"
#include <stdexcept>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define CUDACALL_CHECK(call){                             \
    cudaError_t cudaErr = (call);                                              \
    if (cudaErr != cudaSuccess) {                                              \
        std::cerr << "CUDA Error: " << cudaGetErrorString(cudaErr)             \
                    << " (code " << cudaErr << ") at " << __FILE__             \
                    << ":" << __LINE__ << std::endl;                       \
        exit(EXIT_FAILURE);                                                \
    }                                                                           \
}

namespace IdioticML {

/**
 * Check if there exists an NVIDIA GPU or not
 */
bool CUDA::is_GPUreadyToUse()
{
    int gpuCount = 0;

    cudaError_t err = cudaGetDeviceCount(&gpuCount);
    if (err != cudaSuccess){
        throw std::runtime_error(cudaGetErrorString(err));
    }

    if (gpuCount > 0){
        return false;
    }
    return false;
}



// Generic kernel for adding two tensors
template<typename T>
__global__ void tensorAdditionKernel(const T *d_tensor1, 
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
    unsigned int numBlocks = totalEntries / THREADSPERBLOCK + 1;

    // Allocate and copy data to GPU
    T *d_tensor1;
    T *d_tensor2;
    T *d_tensor;

    size_t size = totalEntries * sizeof(T);
    CUDACALL_CHECK(cudaMalloc(&d_tensor1, size));
    CUDACALL_CHECK(cudaMalloc(&d_tensor2, size));
    CUDACALL_CHECK(cudaMalloc(&d_tensor, size));
    CUDACALL_CHECK(cudaMemcpy(d_tensor1, h_tensor1, size, cudaMemcpyHostToDevice));
    CUDACALL_CHECK(cudaMemcpy(d_tensor2, h_tensor2, size, cudaMemcpyHostToDevice));

    // Launch the kernel
    tensorAdditionKernel<T><<<numBlocks, THREADSPERBLOCK>>>(d_tensor1, 
                                                                  d_tensor2, 
                                                                  d_tensor, 
                                                                  totalEntries);
    cudaDeviceSynchronize();
    
    // Copy the result 
    CUDACALL_CHECK(cudaMemcpy(h_tensor, d_tensor, size, cudaMemcpyDeviceToHost));

    // Free the pointer in GPU
    cudaFree(d_tensor1);
    cudaFree(d_tensor2);
    cudaFree(d_tensor);
}



// Generic kernel for subtracting two tensors
template<typename T>
__global__ void tensorSubtractionKernel(const T *d_tensor1, 
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
    unsigned int numBlocks = totalEntries / THREADSPERBLOCK + 1;

    // Allocate and copy data to GPU
    T *d_tensor1;
    T *d_tensor2;
    T *d_tensor;

    size_t size = totalEntries * sizeof(T);
    CUDACALL_CHECK(cudaMalloc(&d_tensor1, size));
    CUDACALL_CHECK(cudaMalloc(&d_tensor2, size));
    CUDACALL_CHECK(cudaMalloc(&d_tensor, size));
    CUDACALL_CHECK(cudaMemcpy(d_tensor1, h_tensor1, size, cudaMemcpyHostToDevice));
    CUDACALL_CHECK(cudaMemcpy(d_tensor2, h_tensor2, size, cudaMemcpyHostToDevice));

    // Launch the kernel
    tensorSubtractionKernel<T><<<numBlocks, THREADSPERBLOCK>>>(d_tensor1, 
                                                                     d_tensor2, 
                                                                     d_tensor, 
                                                                     totalEntries);
    cudaDeviceSynchronize();
    
    // Copy the result 
    CUDACALL_CHECK(cudaMemcpy(h_tensor, d_tensor, size, cudaMemcpyDeviceToHost));

    // Free the pointer in GPU
    cudaFree(d_tensor1);
    cudaFree(d_tensor2);
    cudaFree(d_tensor);
}


template<typename T>
__global__
void matrixMultKernel(const T* d_matrix1,
                      const T* d_matrix2,
                      T *d_matrix,
                      unsigned int commonSize,  // In matrix multiplication, the first operand's columns must be the same with the second operand rows
                      unsigned int rows,    // The number of rows of the result matrix
                      unsigned int cols)    // The number of columns of the result matrix
{
    
    unsigned int colIdx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;

    if (rowIdx < rows && colIdx < cols){
        T buffer = 0;
        for (unsigned int i = 0; i < commonSize; i++){
            buffer += ( d_matrix1[rowIdx * commonSize + i] * d_matrix2[colIdx + cols * i] );
        }
        d_matrix[rowIdx * cols + colIdx] = buffer;
    }

}
template<typename T>
void CUDA::KernelWrap<T>::matrixMultiplication(const T *h_matrix1,
                                               const std::vector<unsigned int>& h_matrix1_Dim,
                                               const T *h_matrix2,
                                               const std::vector<unsigned int>& h_matrix2_Dim,
                                               T *h_matrix)
{
    T *d_matrix1;
    T *d_matrix2;
    T *d_matrix;

    // rows and columns of the result matrix
    unsigned int rows = h_matrix1_Dim[1];
    unsigned int cols = h_matrix2_Dim[0];
    
    // Calculate the total number of entries of each given matrix.
    size_t matrix1_size = h_matrix1_Dim[0] * h_matrix1_Dim[1] * sizeof(T);
    size_t matrix2_size = h_matrix2_Dim[0] * h_matrix2_Dim[1] * sizeof(T);
    size_t matrix_size = rows * cols * sizeof(T);

    // Allocate and copy matrices' entries to the GPU
    CUDACALL_CHECK(cudaMalloc(&d_matrix1, matrix1_size));
    CUDACALL_CHECK(cudaMalloc(&d_matrix2, matrix2_size));
    CUDACALL_CHECK(cudaMalloc(&d_matrix, matrix_size));
    CUDACALL_CHECK(cudaMemcpy(d_matrix1, h_matrix1, matrix1_size, cudaMemcpyHostToDevice));
    CUDACALL_CHECK(cudaMemcpy(d_matrix2, h_matrix2, matrix2_size, cudaMemcpyHostToDevice));


    unsigned int threadsPerDim = (unsigned int)std::sqrt(THREADSPERBLOCK);
    dim3 dimBlock(threadsPerDim, threadsPerDim); // Number of threads
    dim3 dimGrid(cols / threadsPerDim + 1, rows / threadsPerDim + 1); // Number of blocks

    //Launch the kernel
    matrixMultKernel<T><<<dimGrid, dimBlock>>>(d_matrix1,  
                                               d_matrix2, 
                                               d_matrix, 
                                               h_matrix1_Dim[0], 
                                               rows, 
                                               cols);
    cudaDeviceSynchronize();  // Wait for kernel to finish

    cudaMemcpy(h_matrix, d_matrix, matrix_size, cudaMemcpyDeviceToHost);

    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_matrix);
}





// I find this god damn ridiculous
template class CUDA::KernelWrap<int>;
template class CUDA::KernelWrap<float>;
template class CUDA::KernelWrap<double>;


}