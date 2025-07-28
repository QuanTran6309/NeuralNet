#include "operations/cpu_backend.hpp"

#define NOCUDA
#include "utils/logger.cuh"

namespace IdioticML{


template<typename T>
inline void additionHelper(void *dest, const void *src1, const void *src2, unsigned int numberOfEntries){
    for (unsigned int i = 0; i < other.getTotalEntries(); i++){
        ((T *)dest)[i] = ((T *)this->tensorPtr)[i] + ((T *)otherLin.tensorPtr)[i];
    }
}
void CPU_Backend::add(void *dest, 
                              const void *src1, 
                              const void *src2, 
                              unsigned int numberOfEntries, 
                              const TensorType& type) 
{
    // This is ridiculous but I have no better way.
    switch (type)
    {
    case TensorType::FLOAT: {
        additionHelper<float>(dest, src1, src2, numberOfEntries);
        break;
    }
    case TensorType::DOUBLE: {
        additionHelper<float>(dest, src1, src2, numberOfEntries);
        break;
    }
    default:
        LOGEXCEPTION("Unknown datatype");
    }
}   

void CPU_Backend::mult(){
    
}


}