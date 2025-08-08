#ifndef SEQUENCE
#define SEQUENCE

#include "algebra/tensor.hpp"
#include <vector>

namespace IdioticML{

enum ActFuncPool {
    Relu,
    Sigmoid
};

struct Layer
{
    unsigned int nodes;
    ActFuncPool actFunc;
};


struct Sequence
{
private:

public:
    Sequence(const std::vector<Layer>& layers){
        
    }
    

    Tensor operator()(const Tensor& input){

    }
};


}
#endif