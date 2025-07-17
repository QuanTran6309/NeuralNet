#ifndef CUBLAS_SINGLETON
#define CUBLAS_SINGLETON

#include <cuda_runtime.h>
#include <cublas_v2.h>

class CublasSingleton {
private:
    static CublasSingleton *instance;
    static unsigned int useCounter;
    cublasHandle_t handler;
    CublasSingleton();
    ~CublasSingleton();

public:

    static const cublasHandle_t& getHandler();

    static void releaseHandler();

};


#endif