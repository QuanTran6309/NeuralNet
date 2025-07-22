
#ifndef MATRIX
#define MATRIX
#include "tensor.hpp"
#include <iostream>

namespace IdioticML {

class Matrix : public Tensor{
private:
    /**
     * DANGEROUS constructor.
     * 
     * This constructor is primarily used in Matrix addition and subtraction. This
     * constructor will technically take the ownership of the 
     */
    Matrix(Tensor& other);

public:
    Matrix() = default;

    // Init an empty matrix
    Matrix(unsigned int columns,
           unsigned int rows,
           void *src_tensor = nullptr,
           DataType::DataType type = DataType::DataType::FLOAT,
           bool isOnGPU = false); 

    unsigned int rows() const;
    unsigned int cols() const;

    // CPU matrix cross product.
    Matrix operator*(const Matrix& other) const;

    
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    
};
}

#endif

