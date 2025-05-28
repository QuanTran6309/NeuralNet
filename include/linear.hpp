#ifndef LINEAR
#define LINEAR


#include "tensor.hpp"
#include <vector>



template<typename T>
class Linear{
    static_assert(std::is_arithmetic<T>::value, "Only numeric values");
private:
    Tensor<T> inLayer;
    Tensor<T> outLayer;
    Tensor<T> weights;
    Tensor<T> biases;
    T (* actFunc)(T);
public:
    Linear(unsigned int in_nodes, unsigned int out_nodes, T (* actFunc)(T) = nullptr){
        this->inLayer = Tensor<T>({1, in_nodes});
        this->outLayer = Tensor<T>({1, out_nodes});
        this->weights = Tensor<T>({in_nodes, out_nodes});
        this->biases = Tensor<T>({1, out_nodes});

        // Activation function
        this->actFunc = actFunc;
    }

    // Proceed the computation of weights and input tensor
    void compute(const Tensor<T> prevOut){
        if (prevOut.getDim() != this->inLayer.getDim()){
            throw std::runtime_error("In and Out dimension mismatch");
        }
        this->inLayer = prevOut;

        // Traditional maxtrix vector cross product algorithm
        std::vector<unsigned int> weightsDim = this->weights.getDim();
        T *outVec = new T[weightsDim[1]]; // Used to set the value for the output tensor
        for (unsigned int i = 0; i < weightsDim[1]; i++){
            T buffer = 0;
            for (unsigned int j = 0; j < weights[0]; j++){
                buffer += (this->weights.getEntry({j, i}) * this->inLayer.getEntry({0, j}));
            }
            outVec[i] = buffer + this->biases.getEntry({0, i});
            if (this->actFunc){
                outVec[i] = this->actFunc(outVec[i]);
            }
        }

        this->outLayer.setTensor(outVec, weightsDim[1]);
    }

    

    Tensor<T> getOutPut(){
        return this->outLayer;
    }

};



#endif


