#ifndef NEURALNET
#define NEURALNET
#include "algebra/tensor.hpp"
#include <vector>
namespace IdioticML {



struct NeuralNet {
private:
    std::vector<Tensor> layers;
public:
    NeuralNet(const std::vector<unsigned int>& layers);
};
/*
NeuralNet({90, none},
          {100, Relu},
          {9000, Sigmoid}
)
*/



}
#endif
