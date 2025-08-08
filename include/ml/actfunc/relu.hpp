#ifndef RELU
#define RELU
#include "actfunc.hpp"
namespace IdioticML {


class Relu : public ActFunc {

public:
    Relu() = default;
    void compute(Tensor& input) override;
};

}
#endif