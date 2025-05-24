#ifndef WEIGHTS
#define WEIGHTS


#include "tensor.cc"
template<typename T>
class Weights : public Tensor<T>{

private:

public:
    TensorType getType() override {
        return TensorType::Weights;
    }
    Weights(unsigned int prevTotalEntries, 
            unsigned int nextTotalEntries) : Tensor<T>({prevTotalEntries, nextTotalEntries}){}
};

#endif