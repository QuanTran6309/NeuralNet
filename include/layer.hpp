#ifndef LAYER 
#define LAYER


#include "matrix.hpp"
#include "utils.hpp"
#include <vector>


class Layer{

private:
    Matrix<float> inLayer;
    Matrix<float> outLayer;
    Matrix<float> weights;
    Matrix<float> biases;
    float (* actFunc)(float); // Function pointer to activation function in /include/actfunc.hpp
public:
    Layer() = default;

    Layer(unsigned int in_nodes, unsigned int out_nodes, float(* actFunc)(float) = nullptr){
        this->inLayer = Matrix<float>(1, in_nodes);
        this->outLayer = Matrix<float>(1, out_nodes);
        this->weights = Matrix<float>(in_nodes, out_nodes);
        this->biases = Matrix<float>(1, out_nodes);
        
        // Activation function
        this->actFunc = actFunc;
    }

    
    // Randomly assign value to all entries of the tensor.
    void randomlyInit(){
        unsigned int weightsEntries = this->weights.getTotalEntries();
        unsigned int biasesEntries = this->biases.getTotalEntries();
        float *weightsBuf = new float[weightsEntries];
        float *biasesBuf = new float[biasesEntries];

        // Assign random values to the buffer
        for (unsigned int i = 0; i < weightsEntries; i++){
            weightsBuf[i] = Utils::getRealRandom(-100, 100);
            if (i < biasesEntries){
                biasesBuf[i] = Utils::getRealRandom(-100, 100);
            }
        }

        // Set the Weights and Biases Matrices.
        this->weights.setTensor(weightsBuf, weightsEntries);
        this->biases.setTensor(biasesBuf, biasesEntries);

        delete[] weightsBuf;
        delete[] biasesBuf;
    }

    // Proceed the computation of weights and input tensor
    void compute(const Matrix<float>prevOut){
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

    Matrix<float> getWeights(){
        return this->weights;
    }

    Matrix<float> getBiases(){
        return this->biases;
    }

    Matrix<float> getInLayer(){
        return this->inLayer;
    }

    Matrix<float> prevOut(){
        return this->outLayer;
    }
};



#endif


