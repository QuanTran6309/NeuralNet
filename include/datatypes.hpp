#ifndef TENSORTYPES
#define TENSORTYPES

#include <stdexcept>


namespace IdioticML {

namespace DataType{

enum DataType {
    FLOAT,
    DOUBLE
};

inline unsigned int getDatTypeSize(DataType type){
    switch (type)
    {
    case DataType::FLOAT:
        return sizeof(float);    
    case DataType::DOUBLE:
        return sizeof(double);
    default:
        throw std::runtime_error("Unsupported data type");
    }
}

enum DEVICE { GPU, CPU };

}
}

#endif