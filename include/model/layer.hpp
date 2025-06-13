#ifndef LAYER 
#define LAYER

#include <type_traits>
#include "matrix.hpp"

template<typename T>
struct Layer{
    static_assert(std::is_arithmetic<T>::value, "Template parameter must be a primitive type");

    // Function pointer to activation function in /include/actfunc.hpp
    // Or any function that return a numeric value
    T (* actFunc)(T);
    // Weights matrix
    Matrix<T> weight;
    // Bias matrix : a column vector
    Matrix<T> bias;
    // Output matrix, technically a column vector
    Matrix<T> output;

    Layer(unsigned int in_nodes, unsigned int out_nodes, T (* actFunc)(T) = nullptr){
        if (in_nodes == 0 || out_nodes == 0){
            throw std::runtime_error("The number of input and output nodes must be greater than 0");
        }
        this->weight = Matrix<T>(in_nodes, out_nodes);
        this->bias = Matrix<T>(1, out_nodes);
        this->actFunc = actFunc;
    }

    unsigned int in_nodes() const {
        return this->weight.cols();
    }
    unsigned int out_nodes() const {
        return this->weight.rows();
    }
};

#endif


