#ifndef OPSBACKEND
#define OPSBACKEND

#include "algebra/tensor.hpp"

namespace IdioticML {

/**
 * Handle arithmetic operations of Tensor
 * 
 * None of this class methods will check for any conditions of any kinds before performing
 * It plainly does the operations.
 * The class in charged of doing the checking is LinAlg
 */
class OpsBackend {
public:
    OpsBackend() = default;
    virtual ~OpsBackend() = default;
    
    void allocate(void **ptr);
    void deallocate(void **ptr);

    virtual void add(void *dest, 
                     const void *src1, 
                     const void *src2, 
                     unsigned int numberOfEntries, 
                     const TensorType& type) = 0;
    virtual void mult() = 0;
};


}


#endif