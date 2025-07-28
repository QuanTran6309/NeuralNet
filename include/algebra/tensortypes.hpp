#ifndef TENSORTYPES
#define TENSORTYPES

#include <stdexcept>


namespace IdioticML {


enum TensorType {
    FLOAT,
    DOUBLE
};

inline unsigned int getTensorTypeSize(TensorType type){
    switch (type)
    {
    case TensorType::FLOAT:
        return sizeof(float);    
    case TensorType::DOUBLE:
        return sizeof(double);
    default:
        throw std::runtime_error("Unsupported data type");
    }
}

enum DeviceType { GPU, CPU };


}

#endif