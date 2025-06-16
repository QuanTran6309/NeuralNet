
#ifndef TENSOR_TYPES
#define TENSOR_TYPES

#include <unordered_map>


namespace IdioticML {

enum TensorType {
    INT,
    LONGINT,
    UINT,
    UINT8,
    FLOAT,
    DOUBLE
};

const std::unordered_map<TensorType, size_t> TensorTypeSize = {
    {TensorType::INT, sizeof(int)},
    {TensorType::LONGINT, sizeof(long int)},
    {TensorType::UINT, sizeof(unsigned int)},
    {TensorType::UINT8, sizeof(char)},
    {TensorType::FLOAT, sizeof(float)},
    {TensorType::DOUBLE, sizeof(double)}
};

};


#endif

