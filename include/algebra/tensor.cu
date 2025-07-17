#include "tensor.hpp"
#include "utils.hpp"
#include "vendor/cublas.cuh"
#include "vendor/logger.cuh"

#include <cstring>
#include <iostream>
#include <cuda_runtime.h>


namespace IdioticML {

// Translate the position vector to the offset on the memory
unsigned int Tensor::posVecToIndex(const std::vector<unsigned int>& pos, 
                                   const std::vector<unsigned int>& dimensions, 
                                   unsigned int totalEntries) 
{
    if (pos.size() > dimensions.size() || pos.size() == 0){
        throw std::runtime_error("Position vector is invalid");
    }
    unsigned int prevSize = 1;
    unsigned int index = 0;
    for (unsigned int i = 0; i < pos.size(); i++){
        index += (pos[i] * prevSize);
        prevSize *= dimensions[i];
    }
    if (index >= totalEntries){
        throw std::runtime_error("Index out of bound");
    }
    return index;
}


/**
 * Constructor using pointer
 * 
 * @param dimensions: represent the size of each dimension of a tensor. The index corresponds to the level of dimension.
 * @param src_tensor: holding the source that this instance of Tensor should copy from.
 * @param type: data type of each entry of the Tensor.
 */
Tensor::Tensor (const std::vector<unsigned int>& dimensions,
                void *src_tensor,
                DataType::DataType type,
                bool isOnGPU)
{
    this->dimensions = dimensions;
    this->totalEntries = std::accumulate(this->dimensions.begin(), this->dimensions.end(), 1, std::multiplies<unsigned int>());
    if (this->totalEntries == 0){
        throw std::runtime_error("One of the size of the given dimensions is 0.");
    }
    this->type = type;
    this->entrySize = DataType::getDatTypeSize(this->type);
    this->isOnGPU = isOnGPU;
    
    // The total number of bytes this tensor has.
    unsigned int size = this->totalEntries * this->entrySize;

    // Very crucial for the destructor;
    this->tensorPtr = nullptr;

    // Tensor memory allocation depends on if the tensor is on GPU or CPU
    if (this->isOnGPU){
        if (src_tensor != nullptr) {
            CUDA_CHECK(cudaMalloc(&this->tensorPtr, size));
            CUDA_CHECK(cudaMemcpy(this->tensorPtr, src_tensor, size, cudaMemcpyHostToDevice));
        }
    }
    else {
        if (src_tensor != nullptr){
            this->tensorPtr = (char *)std::malloc(size);
            memcpy(this->tensorPtr, src_tensor, size);
        }
    }
}

/**
 * Special and DANGEROUS constructor.
 * 
 * This constructor technically is doing: thisTensor = anotherTensor;
 * The tensor pointer of thisTensor is plainly assigned by anotherTensor's tensor pointer.
 * 
 * I implement this constructor for the sake of convenience when I need to overload the + and - 
 * of the Matrix class. Those two operators of Matrix class work the same way the Tensor class does.
 */
Tensor::Tensor(Tensor& other) : Tensor(other.dimensions, 
                                       nullptr, 
                                       other.type, 
                                       other.isOnGPU)
{
    this->tensorPtr = other.tensorPtr;
    other.tensorPtr = nullptr;
}


/**
 * Destructor.
 * Free the pointer and check if there is any instance of Tensor left, if there is not, destroy the cublas handler.
 */
Tensor::~Tensor(){
    if (this->tensorPtr == nullptr){
        return;
    }
    if (this->isOnGPU){
        CUDA_CHECK(cudaFree(this->tensorPtr));
    }
    else {
        std::free(this->tensorPtr);
    }
}


/**
 * Get the value of a specific entry given by posVec
 * 
 * @param posVec: a vector specifying the position of the entry in the tensor
 * @param buffer: a buffer to hold the return value.
 */
void Tensor::at(std::vector<unsigned int> posVec, 
                void *buffer)
{
    unsigned int index = Tensor::posVecToIndex(posVec, 
                                               this->dimensions, 
                                               this->totalEntries);
    if (this->isOnGPU){
        CUDA_CHECK(cudaMemcpy(buffer, this->tensorPtr + index * this->entrySize, this->entrySize, cudaMemcpyDeviceToHost));
    }
    else {
        memcpy(buffer, this->tensorPtr + index * this->entrySize, this->entrySize);
    }
}


void Tensor::to(DataType::DEVICE device){
    if ((device == DataType::DEVICE::GPU) ^ this->isOnGPU){
        return;
    }
    this->isOnGPU = device == DataType::DEVICE::GPU;

    size_t tensor_nbytes = this->totalEntries * this->entrySize;
    char *tensorBuff;
    if (this->isOnGPU){
        // Moving from CPU to GPU
        CUDA_CHECK(cudaMalloc(&tensorBuff, tensor_nbytes));
        CUDA_CHECK(cudaMemcpy(tensorBuff, this->tensorPtr, tensor_nbytes, cudaMemcpyHostToDevice));
        std::free(this->tensorPtr);
    }
    else {
        // Move from GPU to CPU
        tensorBuff = (char *)std::malloc(tensor_nbytes);
        CUDA_CHECK(cudaMemcpy(tensorBuff, this->tensorPtr, tensor_nbytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(this->tensorPtr));
    }

    this->tensorPtr = tensorBuff;
}

DataType::DEVICE Tensor::device() const{
    return this->isOnGPU ? DataType::DEVICE::GPU : DataType::DEVICE::CPU;
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
    void *bufferPtr = nullptr;
    if (this->isOnGPU){
        size_t tensor_nbytes = this->totalEntries * this->entrySize;
        bufferPtr = std::malloc(tensor_nbytes);
        CUDA_CHECK(cudaMemcpy(bufferPtr, this->tensorPtr, tensor_nbytes, cudaMemcpyDeviceToHost));
    }
    else {
        bufferPtr = this->tensorPtr;
    }

    std::string buffer;

    switch (this->type)
    {
    case DataType::DataType::FLOAT:
        buffer = toStringHelper<float>((float*)bufferPtr, this->dimensions, this->dimensions.size() - 1, 0, 0);
        break;
    case DataType::DataType::DOUBLE:
        buffer = toStringHelper<double>((double*)bufferPtr, this->dimensions, this->dimensions.size() - 1, 0, 0);
        break;
    default:
        throw std::runtime_error("Unsupported datatype");
    }
    
    if (this->isOnGPU){
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
DataType::DataType Tensor::getType() const{
    return this->type;
}


/**
 * Get a specific portion of the tensor.
 * Imagine having a rubik but you just want to remove a middle layer.
 */
Tensor Tensor::operator()(std::vector<Range> bounds){
    if (bounds.size() > this->dimensions.size()){
        throw std::runtime_error("Bounds' dimension exceeds tensor's dimension");
    }

    // Used to store the indices for each entry that we'll copy from source tensor
    std::vector<unsigned int> indices;
    
    // Get the new dimension for the new tensor
    std::vector<unsigned int> newDim;

    // To iterate a tensor in one dimension, we have to use [index * (product of all previous dimension size)]
    unsigned int prevSize = 1;
    for (unsigned int i = 0; i < this->dimensions.size() - 1; i++){
        prevSize *= this->dimensions[i];
    }
    
    /**
     * For example: bounds = {
     *                        {1:3}
     *                        {0:2}
     *                        {2:4} 
     *                       }
     * The loop will convert that to
     *              {
     *              {1, 2, 3},
     *              {0, 1, 2} * 0th dim,
     *              {2, 3, 4} * 0th dim * 1th dim
     *              }
     * and finally perform cartesian product and sum all of elements within each set to get the set
     * of indices to copy from
     *
     * We need to iterate from the bottom of the bounds to ensure the ascending order of the offsets
     */
    for (int i = bounds.size() - 1; i >= 0; i--){
        Range bound = bounds[i];
        std::vector<unsigned int> index;

        // 1:4 get all entries from 1 to 4
        if (bound.start <= bound.end){
            if (bound.end >= this->dimensions[i]){
                throw std::runtime_error("Index out of bound");
            }
            unsigned int dimSize = bound.end - bound.start + 1;

            // Insert the size for this dimension to newDim
            newDim.insert(newDim.begin(), dimSize);
            
            // Insert indices 
            for (unsigned int i = bound.start; i <= bound.end; i++){
                index.push_back(i * prevSize);
            }
        }
        // 4:1 get all entries from 0->1 and from 4->end
        else {
            if (bound.start >= this->dimensions[i]){
                throw std::runtime_error("Index out of bound");
            }
            unsigned int dimSize = bound.end + 1 + (this->dimensions[i] - bound.start);

            // Insert the size for this dimension to newDim
            newDim.insert(newDim.begin(), dimSize);

            // Insert indices 
            for (unsigned int j = 0; j <= bound.end; j++){
                index.push_back(j * prevSize);
            }
            for (unsigned int j = bound.start; j < this->dimensions[i]; j++){
                index.push_back(j * prevSize);
            }
        }
        
        if (i > 0){
            prevSize /= this->dimensions[i - 1];
        }
        
        // Process the indices
        if (indices.size() != 0){
            indices = Utils::sumOfCartesianProd({indices, index});
        }
        else{
            indices = index;
        }
    }

    Tensor new_tensor(newDim, nullptr, this->type, this->isOnGPU);

    // The size in bytes of the new tensor
    unsigned int newTensorSize = this->entrySize * indices.size();
    char *newTensorPtr = nullptr;

    if (this->isOnGPU){
        CUDA_CHECK(cudaMalloc(&newTensorPtr, newTensorSize));
        for (unsigned int i = 0; i < indices.size(); i++){
            CUDA_CHECK(cudaMemcpy(newTensorPtr + (i * this->entrySize), this->tensorPtr + (indices[i] * this->entrySize), this->entrySize, cudaMemcpyDeviceToDevice));
        }
    }
    else {
        newTensorPtr = static_cast<char *>(std::malloc(newTensorSize));
        for (unsigned int i = 0; i < indices.size(); i++){
            memcpy(newTensorPtr + (i * this->entrySize), this->tensorPtr + (indices[i] * this->entrySize), this->entrySize);
        }
    }
    
    new_tensor.tensorPtr = newTensorPtr;
    return new_tensor;
}

/**
 * Overload the + operator.
 * 
 * This operator overload has two helpers: CPUtensorAdd_helper() and GPUtensorAdd_helper()
 */
inline void CPUtensorAdd_helper(void *dest_tensor, 
                                const void *src_tensor1, 
                                const void *src_tensor2,
                                unsigned int numberOfEntries,
                                DataType::DataType datatype)
{
    switch (datatype)
    {
    case DataType::DataType::FLOAT: {
        for (unsigned int i = 0; i < numberOfEntries; i++){
            ((float *)dest_tensor)[i] = ((float *)src_tensor1)[i] + ((float *)src_tensor2)[i];
        }
        break;
    }
    case DataType::DataType::DOUBLE: {
        for (unsigned int i = 0; i < numberOfEntries; i++){
            ((double *)dest_tensor)[i] = ((double *)src_tensor1)[i] + ((double *)src_tensor2)[i];
        }
        break;
    }
    default:
        break;
    }
}

inline void GPUtensorAdd_helper(void *dest_tensor, 
                                const void *other_tensor, 
                                unsigned int numberOfEntries,
                                DataType::DataType datatype)
{
    switch (datatype)
    {
    case DataType::DataType::FLOAT: {
        float alpha = 1;
        cublasSaxpy(CublasSingleton::getHandler(), numberOfEntries, &alpha, (float *)other_tensor, 1, (float *)dest_tensor, 1);
        break;
    }
    case DataType::DataType::DOUBLE: {
        double alpha = 1;
        cublasDaxpy(CublasSingleton::getHandler(), numberOfEntries, &alpha, (double *)other_tensor, 1, (double *)dest_tensor, 1);
        break;
    }
    default:
        break;
    }
    CublasSingleton::releaseHandler();
}

Tensor Tensor::operator+(const Tensor& other) const{
    if (this->dimensions != other.getDim()){
        throw std::runtime_error("Cannot perform addition on two different tensors");
    }
    if (this->isOnGPU ^ other.isOnGPU){
        throw std::runtime_error("Both tensors must be on the same device");
    }

    if (this->type != other.type){
        throw std::runtime_error("Both tensors must have the same data type");
    }

    Tensor newTensor(this->dimensions, 
                     nullptr, 
                     this->type, 
                     this->isOnGPU);

    void *newTensorPtr; // The pointer of the newTensor will later take the ownership of this pointer.
    unsigned int tensor_nbytes = this->entrySize * this->totalEntries;
    
    if (this->isOnGPU){ // Prioritize using GPU rather than CPU
        CUDA_CHECK(cudaMalloc(&newTensorPtr, tensor_nbytes));
        // This is weird, but necessary because the cublas addition method only take two pointers and 
        // it will override the result to the second one.
        CUDA_CHECK(cudaMemcpy(newTensorPtr, other.tensorPtr, tensor_nbytes, cudaMemcpyDeviceToDevice));
        GPUtensorAdd_helper(newTensorPtr, this->tensorPtr, this->totalEntries, this->type);
    }
    else {
        newTensorPtr = std::malloc(this->entrySize * this->totalEntries);
        CPUtensorAdd_helper(newTensorPtr, this->tensorPtr, other.tensorPtr, this->totalEntries, this->type);
    }    

    newTensor.tensorPtr = (char *)newTensorPtr;
    return newTensor;
}



/**
 * Overload the - operator.
 * 
 * This operator overload has two helpers: CPUtensorSub_helper() and GPUtensorSub_helper()
 */
inline void CPUtensorSub_helper(void *dest_tensor, 
                                const void *src_tensor1, 
                                const void *src_tensor2,
                                unsigned int numberOfEntries,
                                DataType::DataType datatype)
{
    switch (datatype)
    {
    case DataType::DataType::FLOAT: {
        for (unsigned int i = 0; i < numberOfEntries; i++){
            ((float *)dest_tensor)[i] = ((float *)src_tensor1)[i] - ((float *)src_tensor2)[i];
        }
        break;
    }
    case DataType::DataType::DOUBLE: {
        for (unsigned int i = 0; i < numberOfEntries; i++){
            ((double *)dest_tensor)[i] = ((double *)src_tensor1)[i] - ((double *)src_tensor2)[i];
        }
        break;
    }
    default:
        break;
    }
}

inline void GPUtensorSub_helper(void *dest_tensor, 
                                const void *other_tensor, 
                                unsigned int numberOfEntries,
                                DataType::DataType datatype)
{
    switch (datatype)
    {
    case DataType::DataType::FLOAT: {
        float alpha = -1;
        cublasSaxpy(CublasSingleton::getHandler(), numberOfEntries, &alpha, (float *)other_tensor, 1, (float *)dest_tensor, 1);
        break;
    }
    case DataType::DataType::DOUBLE: {
        double alpha = -1;
        cublasDaxpy(CublasSingleton::getHandler(), numberOfEntries, &alpha, (double *)other_tensor, 1, (double *)dest_tensor, 1);
        break;
    }
    default:
        break;
    }
    CublasSingleton::releaseHandler();
}

Tensor Tensor::operator-(const Tensor& other) const{
    if (this->dimensions != other.getDim()){
        throw std::runtime_error("Cannot perform addition on two different tensors");
    }
    if (this->isOnGPU ^ other.isOnGPU){
        throw std::runtime_error("Both tensors must be on the same device");
    }
    if (this->type != other.type){
        throw std::runtime_error("Both tensors must have the same data type");
    }

    Tensor newTensor(this->dimensions, 
                     nullptr, 
                     this->type, 
                     this->isOnGPU);

    void *newTensorPtr; // The pointer of the newTensor will later take the ownership of this pointer.
    unsigned int tensor_nbytes = this->entrySize * this->totalEntries;
    
    if (this->isOnGPU){ // Prioritize using GPU rather than CPU
        CUDA_CHECK(cudaMalloc(&newTensorPtr, tensor_nbytes));
        // This is weird, but necessary because the cublas addition method only take two pointers and 
        // it will override the result to the second one.
        CUDA_CHECK(cudaMemcpy(newTensorPtr, other.tensorPtr, tensor_nbytes, cudaMemcpyDeviceToDevice));
        GPUtensorSub_helper(newTensorPtr, this->tensorPtr, this->totalEntries, this->type);
    }
    else {
        newTensorPtr = std::malloc(this->entrySize * this->totalEntries);
        CPUtensorSub_helper(newTensorPtr, this->tensorPtr, other.tensorPtr, this->totalEntries, this->type);
    }    

    newTensor.tensorPtr = (char *)newTensorPtr;
    return newTensor;
}

}