#ifndef LOGGER
#define LOGGER

#include <iostream>

#ifndef NOCUDA
    #include <cuda_runtime.h>
    #include <cublas_v2.h>


    // Converts cublasStatus_t to string
    inline const char* cublasGetErrorString(cublasStatus_t status) {
        switch (status) {
            case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
            case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
            case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
            default: return "UNKNOWN_CUBLAS_ERROR";
        }
    }

    // Checks cublasStatus_t and reports failure
    inline void checkCublasStatus(cublasStatus_t status, const char* file, int line) {
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cuBLAS Error: " << cublasGetErrorString(status)
                    << " at " << file << ":" << line << std::endl;
            std::exit(EXIT_FAILURE); // or throw exception if you prefer
        }
    }
    inline void checkCudaCall (cudaError_t call, const char *file, int line){
        if (call != cudaSuccess){
            std::cerr << "CUDA error: " << cudaGetErrorString(call) 
                    << " at " << file << ":" << line << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
    // Macro to simplify usage
    #define CUBLAS_ERR_CHECK(status) checkCublasStatus(status, __FILE__, __LINE__)
    // Check the call of cuda, such as cudaMalloc
    #define CUDA_ERR_CHECK(call) checkCudaCall(call, __FILE__, __LINE__)

#endif

#define LOGEXCEPTION(message) \
    std::cerr << message << ": at " << __FILE__ << ":" << __LINE__; \
    std::exit(EXIT_FAILURE); \

#endif