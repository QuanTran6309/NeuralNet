
#ifndef DEVICEMANAGER
#define DEVICEMANAGER
#include <memory>
#include "memory/adapter/device_adapter.hpp"

namespace IdioticML{


struct DeviceContext {
private:
    std::shared_ptr<DeviceAdapter> adapter;

public:
    DeviceContext() = default;
    DeviceContext(const Device& device);

    DeviceType getDeviceType();

    /**
     * Tensor's data migration methods,
     * 
     * Call ones of these two methods to migrate data from CPU to GPU, GPU to CPU, or accross different GPU
     * if you have multiple GPU.
     */
    void changeContextToGPU(void **ptr, size_t num_bytes, int newGPU_id = 0);
    void changeContextToCPU(void **ptr, size_t num_bytes);


    // From this line, all method below are just wrapper around the methods with the same name of DeviceAdapter
    /**
     * Allocate memory for the given pointer. 
     * 
     * @param ptr: pointer to poiter of the data
     * @param num_bytes: the number of bytes to be allocated
     * @param src: the source data to be copy from to ptr. Can be left nullptr, the method wont copy anything.
     * 
     */
    void allocate(void **ptr, size_t num_bytes, const void *src = nullptr);

    /**
     * Deallocate memory for the given pointer. 
     * 
     */
    void deallocate(void **ptr);
    
    /**
     * Copy data from the src to dest.
     * 
     * @param dest: can be on either CPU or GPU. The method will detect its device.
     * @param src:  must be on the same device with the device passed into the constructor.
     */
    void copyTo(void *dest, const void* src, size_t num_bytes);


    // Linear Algebra operations
    void add(void *dest, 
             const void *src1, 
             const void *src2, 
             unsigned int numberOfEntries, 
             const TensorType& type);

    /**
     * Perform multiplication
     * 
     * The order and naming of the arguments of this method is very weird but it fits the 
     * relative order of crucial arguments of cublas<t>gemm() of cublas documentation.
     * 
     * Check out the cublas docs.
     */
    void mult(int m, int n, int k,
              const void *src1,
              const void *src2,
              void *dest,
              const TensorType& type);


};



}


#endif