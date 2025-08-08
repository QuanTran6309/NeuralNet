#ifndef ACTFUNC
#define ACTFUNC

#include "algebra/tensor.hpp"
namespace IdioticML {


class ActFunc {
public:
    virtual ~ActFunc() = 0;

    virtual void compute(Tensor& input) = 0;
};


}
#endif

