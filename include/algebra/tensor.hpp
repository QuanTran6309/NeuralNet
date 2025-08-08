
#ifndef TENSOR
#define TENSOR


#include <vector>
#include <memory>
#include <numeric>

#include "tensortypes.hpp"
#include "memory/adapter/device_adapter.hpp"


namespace IdioticML{


struct Range {
    unsigned int start; // inlcusive
    unsigned int end;   // inclusive
};

class ActFunc;

class Tensor {
    
    friend ActFunc;

private:

protected:
    /**
     * Keeps track of the dimensions of the tensor.
     * 
     * The index of each element tells the order of dimension.
     * 0 => first dimension
     * 1 => second dimension
     * 2 => third dimension
     * ... so on
     */
    std::vector<unsigned int> dimensions;

    void *tensorPtr;  // Keep track of tensor data

    std::shared_ptr<DeviceAdapter> deviceAdapter;

    unsigned int totalEntries;   // Total number of entries
    TensorType type;             // Data type of each entry
    unsigned int entrySize;      // The size of the TensorType type


    /**
     * Dangerous constructor.
     * 
     * Instance created by this constructor will take the given adapter and tensorPtr instead
     * of forming and copying like the public constructor.
     * 
     * @param tensorPtr: the member tensorPtr of the instance will take over the ownership of the data this pointer is holding
     * @param adapter: the member deviceAdapter will share the same instance with this.
     * 
     * I implemented this constructor to support creating instance inside arithmetic operator overload.
     */
    Tensor (const std::vector<unsigned int>& dimensions,
            std::shared_ptr<DeviceAdapter> adapter,
            void * tensorPtr,
            TensorType type = TensorType::FLOAT);
    
public:

    // Virtual destructor for polymorphism
    virtual ~Tensor();
    Tensor() = default;

    /**
     * Constructor using pointer
     * 
     * @param dimensions: represent the size of each dimension of a tensor. The index corresponds to the level of dimension.
     * @param src_tensor: holding the source that this instance of Tensor should copy from.
     * @param type: data type of each entry of the Tensor.
     */
    Tensor (const std::vector<unsigned int>& dimensions,
            const void *src = nullptr,
            TensorType type = TensorType::FLOAT,
            Device device = Device{});

    
    /**
     * Get the value of a specific entry given by posVec
     * 
     * @param posVec: a vector specifying the position of the entry in the tensor
     * @param buffer: a buffer to hold the return value.
     */
    void at(std::vector<unsigned int> posVec, void *buffer);

    /**
     * Tell the tensor to migrate the data to either CPU or GPU.
     */
    void toDevice(DeviceType device = DeviceType::CPU, int id = 0);

    /**
     * Get the device the tensor is on.
     */
    DeviceType getDeviceType() const;

    /**
     * Get the visualized string of the Tensor.
     * Just like when you print out a tensor in Pytorch.
     * 
     * This method is just a wrapper for toStringHeler() which is there is real string is formed.
     * 
     * ISSUE: It does not increment the nested dimension like Pytorch does.
     *        But I dont know how to fix.
     */
    std::string toString();

    // Get the dimensions of the tensor
    std::vector<unsigned int> getDim() const;

    // Get the total number of entries of this tensor, depending on the dimension size
    unsigned int getTotalEntries() const;

    unsigned int getEntrySize() const;

    // Get the data type of each entry of the Tensor
    TensorType getType() const;

    
    /**
     * Get a specific portion of the tensor.
     * Imagine having a rubik but you just want to remove a middle layer.
     */
    Tensor slice(std::vector<Range> bounds);


    Tensor operator+(const Tensor& other);
    Tensor operator*(const Tensor& other);

};
}

#endif
