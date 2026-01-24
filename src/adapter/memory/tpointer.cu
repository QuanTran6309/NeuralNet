#include "adapter/memory/tpointer.hpp"
#include "adapter/memory/cpu_memory.hpp"
#include "adapter/memory/gpu_memory.cuh"
#include "utils/logger.cuh"

namespace IdioticML {


TPointer::TPointer(const Device& device, size_t n_bytes, const void *src) 
                  : counter(new unsigned int(1))

{
    if (n_bytes == 0){
        LOGEXCEPTION("The number of bytes to be allocated must be greater than 0")
    }

    if (device.type == DeviceType::CPU){
        this->memory = std::make_shared<CpuMemory>(device);
    }
    else {
        this->memory = std::make_shared<GpuMemory>(device);
    }
    this->memory->allocate(&this->ptr, n_bytes, src);
}

void TPointer::release(){

    if (!this->counter){
        return;
    }

    // Check the counter, if = 0, free memory
    (*this->counter)--;
    if (*this->counter == 0){
        delete this->counter;
        this->memory->deallocate(&this->ptr);
    }
}


TPointer::~TPointer(){
    this->release();
}


TPointer& TPointer::operator=(const TPointer& other){
    if (this != &other){
        // Because this pointer may be refering to a different chunk of memory
        // So need to release it first.
        this->release();
    
        this->counter = other.counter;
        this->ptr = other.ptr;

        (*this->counter)++;
    }
}




}