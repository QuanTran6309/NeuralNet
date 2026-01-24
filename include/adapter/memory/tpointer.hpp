#ifndef TPOINTER
#define TPOINTER

#include "adapter/memory/device_memory.hpp"
#include <memory>

namespace IdioticML{

/**
 * Tensor pointer
 * 
 * Technically, it works just like a shared pointer.
 */
struct TPointer
{
private:
    void *ptr;
    unsigned int *counter;
    std::shared_ptr<DeviceMemory> memory;

    void release();
public:
    TPointer(const Device& device, size_t n_bytes, const void *src);
    ~TPointer();

    TPointer& operator=(const TPointer& other);
};


}

#endif

