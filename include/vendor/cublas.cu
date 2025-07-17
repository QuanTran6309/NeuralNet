#include "cublas.cuh"
#include "logger.cuh"


CublasSingleton* CublasSingleton::instance = nullptr;
unsigned int CublasSingleton::useCounter = 0;

CublasSingleton::CublasSingleton(){
    CUBLAS_CHECK(cublasCreate(&this->handler));
}

CublasSingleton::~CublasSingleton(){
    CUBLAS_CHECK(cublasDestroy(this->handler));
}

const cublasHandle_t& CublasSingleton::getHandler(){
    if (CublasSingleton::instance == nullptr){
        CublasSingleton::instance = new CublasSingleton();
    }    
    CublasSingleton::useCounter++;
    return CublasSingleton::instance->handler;
}


void CublasSingleton::releaseHandler(){
    if (CublasSingleton::useCounter == 0){
        return;
    }
    CublasSingleton::useCounter--;
    if (CublasSingleton::useCounter == 0){
        delete CublasSingleton::instance;
        CublasSingleton::instance = nullptr;
    }
}





