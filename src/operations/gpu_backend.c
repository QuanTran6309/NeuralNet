#include "operations/gpu_backend.cuh"
#include "utils/logger.cuh"
#include <cuda_runtime.h>


namespace IdioticML{

GPU_Backend::GPU_Backend() {
    cublasCreate(&this->handler);
}
GPU_Backend::GPU_Backend() {
    cublasDestroy(this->handler);
}


void GPU_Backend::add(void *dest, 
                      const void *src1, 
                      const void *src2, 
                      unsigned int numberOfEntries, 
                      const TensorType& type)
{
    // This is quite ridiculous but it is how cublas works.
    // Cublas' method for addition do not write result into a new chunk of memory but
    // overwrite into the second given pointer.
    CUDA_ERR_CHECK(cudaMemcpy(dest, src2, numberOfEntries * getDatTypeSize(type), cudaMemcpyDeviceToDevice));

    switch (type)
    {
    case TensorType::FLOAT:{
        float alpha = 1;
        CUBLAS_ERR_CHECK(cublasSaxpy(this->handler, numberOfEntries, 
                                     &alpha, 
                                     (float *)src1, 1, 
                                     (float *)dest, 1));
        break;
    }
    case TensorType::DOUBLE:{
        double alpha = 1;
        CUBLAS_ERR_CHECK(cublasDaxpy(this->handler, numberOfEntries, 
                                     &alpha, 
                                     (double *)src1, 1, 
                                     (double *)dest, 1));
        break;
    }
    default:
        break;
    }
}




}