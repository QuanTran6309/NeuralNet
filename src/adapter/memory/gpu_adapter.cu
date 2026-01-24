#include "memory/adapter/gpu_adapter.cuh"
#include "utils/logger.cuh"
#include <cuda_runtime.h>


namespace IdioticML{

GPU_adapter::GPU_adapter(const Device& device) : DeviceAdapter(device) {
    if (device.type != DeviceType::GPU){
        LOGEXCEPTION("GPU Adapter requires device of type GPU")
    }

    cublasCreate(&this->handler);
}

GPU_adapter::~GPU_adapter(){
    cublasDestroy(this->handler);
}

void GPU_adapter::allocate(void **ptr, size_t num_bytes, const void *src) {
    CUDA_ERR_CHECK(cudaMalloc(ptr, num_bytes));
    if (src == nullptr){
        return;
    }

    cudaSetDevice(this->device.id);

    cudaPointerAttributes attr;
    CUDA_ERR_CHECK(cudaPointerGetAttributes(&attr, src));
    cudaMemcpyKind cpyKind = cudaMemcpyHostToDevice;

    // If *src is on the GPU, then must use CUDA method
    if (attr.type == cudaMemoryTypeDevice){
        cpyKind = cudaMemcpyDeviceToDevice;
    }
    else if (attr.type == cudaMemoryTypeManaged){
        // If the src pointer is on the unified memory, 
        // we need to make sure any async operations about that chunk of memory
        // is done before reading.
        cudaDeviceSynchronize();
    }

    CUDA_ERR_CHECK(cudaMemcpy(*ptr, src, num_bytes, cpyKind));
}

void GPU_adapter::deallocate(void **ptr) {
    CUDA_ERR_CHECK(cudaFree(*ptr));
    *ptr = nullptr;
}

void GPU_adapter::copyTo(void *dest, const void *src, size_t num_bytes){
    cudaPointerAttributes attr;
    CUDA_ERR_CHECK(cudaPointerGetAttributes(&attr, dest));

    // If *src is on the GPU, then must use CUDA method
    if (attr.type == cudaMemoryTypeDevice){
        CUDA_ERR_CHECK(cudaMemcpy(dest, src, num_bytes, cudaMemcpyDeviceToDevice));
    }
    else {
        if (attr.type == cudaMemoryTypeManaged){
            // If the src pointer is on the unified memory, 
            // we need to make sure any async operations about that chunk of memory
            // is done before reading.
            cudaDeviceSynchronize();
        }
        CUDA_ERR_CHECK(cudaMemcpy(dest, src, num_bytes, cudaMemcpyDeviceToHost));
    }
}



template<typename T>
__global__ void simultAssign(T *dest, 
                            const T *src, 
                            const unsigned int *indices, 
                            unsigned int numberOfIndices)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x < numberOfIndices){
        dest[x] = src[indices[x]];
    }
}
void GPU_adapter::copyAtIndices(void *dest, 
                                const void *src, 
                                const unsigned int *indices, 
                                unsigned int numberOfIndices,
                                const TensorType& type)
{
    int blocks = (numberOfIndices + CHUNK - 1) / CHUNK;
    auto assignFunctor = [&](auto type_t) {
        using T = decltype(type_t);
        simultAssign<T><<<blocks, CHUNK>>>(static_cast<T *>(dest), static_cast<const T *>(src), indices, numberOfIndices);
    };

    type_dispatcher(type, assignFunctor);
}


bool GPU_adapter::isGPU() {return true;}
bool GPU_adapter::isCPU() {return false;}
int GPU_adapter::getGPU_id(){
    return this->device.id;
}


void GPU_adapter::add(void *dest, 
                      const void *src1, 
                      const void *src2, 
                      unsigned int numberOfEntries, 
                      const TensorType& type)
{
    // This is quite ridiculous but it is how cublas works.
    // Cublas' method for addition do not write result into a new chunk of memory but
    // overwrite into the second given pointer.
    CUDA_ERR_CHECK(cudaMemcpy(dest, src2, numberOfEntries * getTensorTypeSize(type), cudaMemcpyDeviceToDevice));

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

void GPU_adapter::mult(int m, int n, int k,
                       const void *src1,
                       const void *src2,
                       void *dest,
                       const TensorType& type) 
{
    switch (type)
    {
    case TensorType::FLOAT:{
        float a = 1;
        float b = 0;
        CUBLAS_ERR_CHECK(cublasSgemm(this->handler,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     m, n, k,
                                     &a,
                                     (float *)src2, m,
                                     (float *)src1, k,
                                     &b,
                                     (float *)dest, m));   
        break;
    }
    case TensorType::DOUBLE:{
        double a = 1;
        double b = 0;
        CUBLAS_ERR_CHECK(cublasDgemm(this->handler,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     m, n, k,
                                     &a,
                                     (double *)src2, m,
                                     (double *)src1, k,
                                     &b,
                                     (double *)dest, m));
        break;
    }
    default:
        LOGEXCEPTION("Unsupported data type")
    }
}




}