#ifndef CPU_WRAP
#define CPU_WRAP

#include "types.hpp"

namespace IdioticML {


class CPU {
private:
public:
    /**
     * Perform tensor addition 
     */
    static void tensorAddition(const void *h_tensor1, 
                               const void *h_tensor2, 
                               void *h_tensor, 
                               TensorType type,
                               unsigned int totalEntries);
    
};

}


#endif