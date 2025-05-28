#ifndef LINEAR
#define LINEAR


#include "matrix.hpp"
#include <vector>



template<typename T>
class Linear{
    static_assert(std::is_arithmetic<T>::value, "Only numeric values");
    
private:
    Matrix<T> inLayer;
    Matrix<T> outLayer;
    Matrix<T> weights;
    Matrix<T> biases;
    T (* actFunc)(T); // Function pointer to activation function in /include/actfunc.hpp

public:
    Linear(unsigned int in_nodes, unsigned int out_nodes, T (* actFunc)(T) = nullptr){
        this->inLayer = Matrix<T>({1, in_nodes});
        this->outLayer = Matrix<T>({1, out_nodes});
        this->weights = Matrix<T>({in_nodes, out_nodes});
        this->biases = Matrix<T>({1, out_nodes});

        // Activation function
        this->actFunc = actFunc;
    }

    // Proceed the computation of weights and input tensor
    void compute(const Matrix<T> prevOut){
        if (prevOut.getDim() != this->inLayer.getDim()){
            throw std::runtime_error("In and Out dimension mismatch");
        }
        this->inLayer = prevOut;

        // These operators are already overloaded.
        // Matrices opeartions.
        this->outLayer = this->weights * this->inLayer + this->biases;
        
        // Apply activation function to all nodes on outLayer
        for (unsigned int i = 0; i < this->outLayer.getTotalEntries(); i++){
            this->outLayer.setEntry({0, i}, this->actFunc(this->outLayer.getEntry({0, i})));
        }
    }

    Matrix<T> getOutPut(){
        return this->outLayer;
    }

};



#endif


