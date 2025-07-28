#include "algebra/linalg.hpp"

#define NOCUDA
#include "utils/logger.cuh"

namespace IdioticML{

LinAlg::LinAlg(Tensor* destTensor, const Tensor& tensor)
    : destTensor(destTensor),
      tensorBufferPtr(tensor.tensorPtr), 
      dimensions(tensor.dimensions),
      isFirstOperation(true), 
      type(tensor.type),
      deviceContext(tensor.deviceContext){}



LinAlg& LinAlg::add(const Tensor& other){
    if (this->destTensor->dimensions != other.dimensions || this->destTensor->type != other.type){
        LOGEXCEPTION("Both tensors must have the same dimensions and entry type to perform addition")
    }
    void *temporaryBuffer = this->destTensor->tensorPtr;

    // Allocate new 
    this->destTensor->deviceContext->allocate(&this->destTensor->tensorPtr, other.entrySize * other.totalEntries);

    // Perform addition
    this->deviceContext->add(this->tensorBufferPtr, 
                             other.tensorPtr, 
                             temporaryBuffer, 
                             other.getTotalEntries(),
                             this->type);

    if (this->isFirstOperation){
        // Do not free the temporaryBuffer in the first operations because it is pointing
        // to the actual tensor given in the constructor, freeing it can cause undefined behavior.
        this->isFirstOperation = false;
    }
    else {
        this->deviceContext->deallocate(&temporaryBuffer);
    }

    return *this;
}

LinAlg& LinAlg::mult(const Tensor& other){
    if (this->dimensions.size() != 2 || other.dimensions.size() != 2){
        LOGEXCEPTION("Only matrices can have multiplication, not tensor.")
    }
    if (this->type != other.type){
        LOGEXCEPTION("Both matrices must have the same entry type")
    }
    if (this->dimensions[0] != other.dimensions[1]){
        LOGEXCEPTION("Dimensions of the given matrices are not appropriate for multiplication")
    }

    void *temporaryBuffer = this->tensorBufferPtr;

    // Allocate new space for the tensor.
    this->deviceContext->allocate(&this->tensorBufferPtr, this->dimensions[1] * other.dimensions[0] * other.getEntrySize());

    // Proceed multiplication
    this->deviceContext->mult(other.dimensions[0], this->dimensions[1], this->dimensions[0],
                              other.tensorPtr,
                              temporaryBuffer,
                              this->tensorBufferPtr);
                              
    // This is multiplication, which means the dimension will change
    this->dimensions[0] = other.dimensions[0];

    if (this->isFirstOperation){
        // Do not free the temporaryBuffer in the first operations because it is pointing
        // to the actual tensor given in the constructor, freeing it can cause undefined behavior.
        this->isFirstOperation = false;
    }
    else {
        this->deviceContext->deallocate(&temporaryBuffer);
    }

    return *this;
}



}