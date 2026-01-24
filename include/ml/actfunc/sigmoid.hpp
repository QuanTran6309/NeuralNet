#ifndef RELU
#define RELU
#include "actfunc.hpp"
namespace IdioticML {


class Sigmoid : public ActFunc {

public:
    Sigmoid() = default;
    void compute(Tensor& input) override;
};

}
#endif