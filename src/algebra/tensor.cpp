#include "algebra/tensor.hpp"
#include "utils/algorithm.hpp"

#include <cstring>
#include <iostream>


namespace IdioticML {

/**
 * Constructor using pointer
 * 
 * @param dimensions: represent the size of each dimension of a tensor. The index corresponds to the level of dimension.
 * @param src_tensor: holding the source that this instance of Tensor should copy from.
 * @param type: data type of each entry of the Tensor.
 */
Tensor::Tensor (const std::vector<unsigned int>& dimensions,
                const void *src,
                TensorType type,
                Device device)
{
    this->dimensions = dimensions;
    this->totalEntries = std::accumulate(this->dimensions.begin(), this->dimensions.end(), 1, std::multiplies<unsigned int>());
    if (this->totalEntries == 0){
        throw std::runtime_error("One of the size of the given dimensions is 0.");
    }
    this->type = type;
    this->entrySize = getDatTypeSize(this->type);
    this->deviceContext = std::make_shared<DeviceContext>(device);
    this->deviceContext->allocate(&this->tensorPtr, this->totalEntries * this->entrySize, src);
}

/**
 * Destructor.
 * Free the pointer and check if there is any instance of Tensor left, if there is not, destroy the cublas handler.
 */
Tensor::~Tensor(){
    if (this->tensorPtr == nullptr){
        return;
    }
    this->deviceContext->deallocate(&this->tensorPtr);
}

/**
 * Get the visualized string of the Tensor.
 * Just like when you print out a tensor in Pytorch.
 * 
 * This method is just a wrapper for toStringHeler() which is there is real string is formed.
 * 
 * ISSUE: It does not increment the nested dimension like Pytorch does.
 *        But I dont know how to fix.
 */
template<typename T>
std::string toStringHelper(T *tensorPtr,
                           const std::vector<unsigned int>& dimensions,
                           unsigned int nth_dim,
                           unsigned int offset,
                           unsigned int depth)
{
    std::string buffer = "";
    if (nth_dim == 0){
        buffer = std::to_string(tensorPtr[offset]);
        for (unsigned int i = 1; i < dimensions[nth_dim]; i++){
            buffer += (", " + std::to_string(tensorPtr[offset + i]));
        }
        return buffer;
    }

    std::vector<std::string> brac_buffer(dimensions[nth_dim]);
    for (unsigned int i = 0; i < brac_buffer.size(); i++){
        // What the fuck?
        brac_buffer[i] = "[" + toStringHelper<T>(tensorPtr, dimensions, nth_dim - 1, i * std::accumulate(dimensions.begin(), dimensions.begin() + nth_dim, 1, std::multiplies<unsigned int>()) + offset, depth + 1) + "]";
    }

    buffer = brac_buffer[0];
    for (unsigned int i = 1; i < brac_buffer.size(); i++){
        buffer += (",\n" + std::string(depth, ' ') + brac_buffer[i]);
    }

    return buffer;
}
std::string Tensor::toString(){
    void *bufferPtr = this->tensorPtr;
    if (this->deviceContext->getDeviceType() != DeviceType::CPU){
        size_t num_bytes = this->entrySize * this->totalEntries;
        bufferPtr = std::malloc(num_bytes);
        this->deviceContext->copyTo(bufferPtr, this->tensorPtr, num_bytes);
    }
    std::string buffer;

    switch (this->type)
    {
    case TensorType::FLOAT:
        buffer = toStringHelper<float>((float*)bufferPtr, this->dimensions, this->dimensions.size() - 1, 0, 0);
        break;
    case TensorType::DOUBLE:
        buffer = toStringHelper<double>((double*)bufferPtr, this->dimensions, this->dimensions.size() - 1, 0, 0);
        break;
    default:
        throw std::runtime_error("Unsupported datatype");
    }
    
    if (this->deviceContext->getDeviceType() != DeviceType::CPU){
        std::free(bufferPtr);
    }
    return buffer;
}


// Get the dimensions of the tensor
std::vector<unsigned int> Tensor::getDim() const{
    return this->dimensions;
}

// Get the total number of entries of this tensor, depending on the dimension size
unsigned int Tensor::getTotalEntries() const {
    return this->totalEntries;
}

// Get the data type of each entry of the Tensor
TensorType Tensor::getType() const{
    return this->type;
}



}