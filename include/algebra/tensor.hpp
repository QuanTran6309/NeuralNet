
#ifndef TENSOR
#define TENSOR


#include <vector>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#include "datatypes.hpp"


namespace IdioticML{


struct Range {
    unsigned int start; // inlcusive
    unsigned int end;   // inclusive
};

class Tensor {
private:

    // Translate the position vector to the offset on the memory
    static unsigned int posVecToIndex(const std::vector<unsigned int>& pos, 
                                      const std::vector<unsigned int>& dimensions, 
                                      unsigned int totalEntries);

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

    /**
     * Keep track of the tensor inside RAM or GPU
     * Use char * for easier iteration.
     */
    char *tensorPtr; 

    /**
     * Tell if this tensor should be allocated on GPU or CPU.
     */
    bool isOnGPU;

    /**
     * The total number of entries this tensor has.
     * 
     * This member does not tell anything about the dimensions or placees of the entries,
     * just the total number of entries.
     */
    unsigned int totalEntries;

    /**
     * Represent the data type of each entry of the tensor.
     */
    DataType::DataType type; 

    /**
     * Size of each entry in byte
     */
    unsigned int entrySize;    

    /**
     * Take the given pointer to be the tensor.
     * 
     * @param src_ptr: pointer to the source tensor. Can either be on RAM or GPU.
     *                 If the current Tensor is on GPU, it will assume this src_ptr is also
     *                 on GPU; and vice versa.
     * ALERT: Be very careful when using this method, because it does not do any copy
     *        but just simply assign the member pointer to the given pointer without 
     *        any check.
     * NOTE: I primarily use this to avoid buffering a huge tensor when I do any tensor's operations.
     */
    void takePtrOwnership(void * src_ptr);
    
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
            void *src_tensor = nullptr,
            DataType::DataType type = DataType::DataType::FLOAT,
            bool isOnGPU = false);

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
    void to(DataType::DEVICE device);

    /**
     * Get the device the tensor is on.
     */
    DataType::DEVICE device();

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
