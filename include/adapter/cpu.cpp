

#include "cpu.hpp"

namespace IdioticML {


void CPU::tensorAddition(const void *h_tensor1, 
                         const void *h_tensor2, 
                         void *h_tensor, 
                         TensorType type,
                         unsigned int totalEntries)
{
    switch (type)
    {
    case TensorType::INT:
        for (unsigned int i = 0; i < totalEntries; i++){
            ((int *)h_tensor)[i] = ((int *)h_tensor1)[i] + ((int *)h_tensor2)[i];
        }    
        return;
    case TensorType::FLOAT:
        for (unsigned int i = 0; i < totalEntries; i++){
            ((float *)h_tensor)[i] = ((float *)h_tensor1)[i] + ((float *)h_tensor2)[i];
        }    
        return;
    default:
        break;
    }
}


}