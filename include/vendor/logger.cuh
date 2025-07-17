#ifndef LOGGER
#define LOGGER

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <execinfo.h>

inline void printStackTrace() {
    const int max_frames = 64;
    void* buffer[max_frames];
    int nptrs = backtrace(buffer, max_frames);
    char** strings = backtrace_symbols(buffer, nptrs);
    std::cerr << "Stack trace:\n";
    for (int i = 0; i < nptrs; i++) {
        std::cerr << "  " << strings[i] << '\n';
    }
    free(strings);
}

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
        printStackTrace();
        std::exit(EXIT_FAILURE); // or throw exception if you prefer
    }
}

// Macro to simplify usage
#define CUBLAS_CHECK(status) checkCublasStatus(status, __FILE__, __LINE__)




inline void checkCudaCall (cudaError_t call, const char *file, int line){
    if (call != cudaSuccess){
        std::cerr << "CUDA error: " << cudaGetErrorString(call) 
                  << " at " << file << ":" << line << std::endl;
        printStackTrace();
        std::exit(EXIT_FAILURE);
    }
}
// Check the call of cuda, such as cudaMalloc
#define CUDA_CHECK(call) checkCudaCall(call, __FILE__, __LINE__)



#endif