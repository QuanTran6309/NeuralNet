#include "algebra/tensor.hpp"
#include "utils/algorithm.hpp"
#include "utils/logger.cuh"
#include "memory/adapter/gpu_adapter.cuh"
#include "memory/adapter/cpu_adapter.hpp"

#include <iostream>
#include <cstring>


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

                : dimensions(dimensions),
                  totalEntries(std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<unsigned int>())),
                  type(type),
                  entrySize(getTensorTypeSize(type))
{
    if (this->totalEntries == 0){
        LOGEXCEPTION("One of the size of the given dimensions is 0.")
    }

    // Form adapter depending on the type of device user want their tensor to be on
    if (device.type == DeviceType::CPU){
        this->deviceAdapter = std::make_shared<CPU_adapter>(device);
    }
    else {
        this->deviceAdapter = std::make_shared<GPU_adapter>(device);
    }

    this->deviceAdapter->allocate(&this->tensorPtr, this->totalEntries * this->entrySize, src);
}

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
Tensor::Tensor(const std::vector<unsigned int>& dimensions,
               std::shared_ptr<DeviceAdapter> adapter,
               void * tensorPtr,
               TensorType type)
               : dimensions(dimensions),
                 totalEntries(std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<unsigned int>())),
                 type(type),
                 entrySize(getTensorTypeSize(type)),
                 deviceAdapter(adapter),
                 tensorPtr(tensorPtr)
{
    if (this->totalEntries == 0){
        LOGEXCEPTION("One of the size of the given dimensions is 0.")
    }
}



/**
 * Destructor.
 * Free the pointer and check if there is any instance of Tensor left, if there is not, destroy the cublas handler.
 */
Tensor::~Tensor(){
    if (this->tensorPtr == nullptr){
        return;
    }
    this->deviceAdapter->deallocate(&this->tensorPtr);
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
    if (this->deviceAdapter->getDeviceType() != DeviceType::CPU){
        size_t num_bytes = this->entrySize * this->totalEntries;
        bufferPtr = std::malloc(num_bytes);
        this->deviceAdapter->copyTo(bufferPtr, this->tensorPtr, num_bytes);
    }
    std::string buffer;

    auto stringFormation = [&](auto type_t){
        using T = decltype(type_t);
        buffer = toStringHelper<T>(static_cast<T *>(bufferPtr), this->dimensions, this->dimensions.size() - 1, 0, 0);
    };

    type_dispatcher(this->type, stringFormation);
   
    if (this->deviceAdapter->getDeviceType() != DeviceType::CPU){
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

DeviceType Tensor::getDeviceType() const {
    return this->deviceAdapter->getDeviceType();
}

/**
 * Migrate a tensor from CPU to GPU or vice versa.
 */
void Tensor::toDevice(DeviceType type, int id){
    void *buffer = this->tensorPtr;
    this->tensorPtr = nullptr;

    std::shared_ptr<DeviceAdapter> newAdapter = (type == DeviceType::CPU)
                                                ? std::static_pointer_cast<DeviceAdapter>(std::make_shared<CPU_adapter>(Device(type, id)))
                                                : std::static_pointer_cast<DeviceAdapter>(std::make_shared<GPU_adapter>(Device(type, id)));

    newAdapter->allocate(&this->tensorPtr, this->entrySize * this->totalEntries, buffer);

    this->deviceAdapter->deallocate(&buffer);
    this->deviceAdapter = newAdapter;
}

// Get a sliced tensor
Tensor Tensor::slice(std::vector<Range> bounds){
    if (bounds.size() > this->dimensions.size()){
        LOGEXCEPTION("Bounds' dimension exceeds tensor's dimension");
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
            indices = Algorithm::sumOfCartesianProd({indices, index});
        }
        else{
            indices = index;
        }
    }

    void *newTensorPtr; // Dont free this pointer because the returned tensor will take ownership of it.
    this->deviceAdapter->allocate(&newTensorPtr, this->entrySize * this->totalEntries);
    this->deviceAdapter->copyAtIndices(newTensorPtr, this->tensorPtr, indices.data(), indices.size(), this->type);

    Tensor newTensor(newDim, this->deviceAdapter, newTensorPtr, this->type);

    return newTensor;
}


Tensor Tensor::operator+(const Tensor& other){
    if (this->dimensions != other.dimensions){
        LOGEXCEPTION("Both tensor must have the same dimension to perform addition")
    }
    if (this->getDeviceType() != other.getDeviceType()){
        LOGEXCEPTION("Both tensor must be on the same device to perform addition")
    }

    void *newTensorPtr;
    this->deviceAdapter->allocate(&newTensorPtr, this->entrySize * this->totalEntries);
    this->deviceAdapter->add(newTensorPtr, this->tensorPtr, other.tensorPtr, this->totalEntries, this->type);

    Tensor newTensor(this->dimensions, this->deviceAdapter, newTensorPtr, this->type);
    return newTensor;
}


Tensor Tensor::operator*(const Tensor& other){
    if (this->dimensions.size() != 2 || other.dimensions.size() != 2){
        LOGEXCEPTION("Multiplication can only be done on matrix only")
    }
    if (this->dimensions[0] != other.dimensions[1]){
        LOGEXCEPTION("The given matrices' dimensions are not fit for doing multiplication")
    }
    if (this->getDeviceType() != other.getDeviceType()){
        LOGEXCEPTION("Both matrices must be on the same device to do multiplication")
    }
    if (this->type != other.type){
        LOGEXCEPTION("Both matrices must have the same data type for each entry")
    }
    
    void *newTensorPtr;
    this->deviceAdapter->allocate(&newTensorPtr, this->entrySize * this->dimensions[1] * other.dimensions[0]);
    this->deviceAdapter->mult(other.dimensions[0], this->dimensions[1], this->dimensions[0],
                              this->tensorPtr,
                              other.tensorPtr,
                              newTensorPtr, 
                              this->type);

    Tensor newTensor({other.dimensions[0], this->dimensions[1]}, this->deviceAdapter, newTensorPtr, this->type);
    return newTensor;
}





}