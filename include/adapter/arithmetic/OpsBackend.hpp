#ifndef OPSBACKEND
#define OPSBACKEND
#include "algebra/tensortypes.hpp"
namespace IdioticML{

class OpsBackend {
public:

    OpsBackend() = default;

    // Arithmetic operations
    // None of these operations will perform any checks before doing the operations
    virtual void add(void *dest, 
                     const void *src1, 
                     const void *src2, 
                     unsigned int numberOfEntries, 
                     const TensorType& type) = 0;

    virtual void mult(int m, int n, int k,
                      const void *src1,
                      const void *src2,
                      void *dest,
                      const TensorType& type) = 0;
};


}
#endif