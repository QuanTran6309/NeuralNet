
#ifndef ACTFUNC
#define ACTFUNC

#include <unordered_map>
#include <functional>
#include "algebra/tensor.hpp"

namespace IdioticML {

enum ActFunc {
    Relu,
    Softmax,
    Sigmoid
};

class ActFuncPool {
    
public:
    static void relu(Tensor& tensor);
    static void softmax(Tensor& tensor);
    static void sigmoid(Tensor& tensor);
};



}

#endif