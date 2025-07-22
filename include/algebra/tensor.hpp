
#ifndef TENSOR
#define TENSOR


#include <vector>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#include "tensortypes.hpp"
#include "memory/device_context.hpp"


namespace IdioticML{


struct Range {
    unsigned int start; // inlcusive
    unsigned int end;   // inclusive
};

class Tensor {

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
    char *tensorPtr;  // Keep track of tensor data
    std::shared_ptr<DeviceContext> deviceContext; // Handle allocate and deallocate tensorPtr. Abstract away the complexity of using GPU and CPU
    unsigned int totalEntries;   // Total number of entries
    TensorType type;             // Data type of each entry
    unsigned int entrySize;      // The size of the TensorType type

    /**
     * Special and DANGEROUS constructor.
     * 
     * This constructor technically is doing: thisTensor = anotherTensor;
     * The tensor pointer of thisTensor is plainly assigned by anotherTensor's tensor pointer.
     * 
     * I implement this constructor for the sake of convenience when I need to overload the + and - 
     * of the Matrix class. Those two operators of Matrix class work the same way the Tensor class does.
     */
    Tensor (Tensor& other);
    
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
     * 
     */
    void to(DeviceType device);

    /**
     * Get the device the tensor is on.
     */
    DeviceType device() const;

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

    // Get the data type of each entry of the Tensor
    TensorType getType() const;

    /**
     * Get a specific portion of the tensor.
     * Imagine having a rubik but you just want to remove a middle layer.
     */
    Tensor operator()(std::vector<Range> bounds);

    /**
     * Overload the + operator.
     */
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
};
}

#endif
