#ifndef DENSE
#define DENSE

#include "layer.hpp"
#include "matrix.hpp"
#include "utils.hpp"
#include <iostream>
#include <string>

/**
 * The Fully connected layer, behave just like a typical neural network.
 * The name "fully connected layer" is too long, so I took the name Dense.
 * 
 * This class does not keep track of the input layer.
 */
template<typename T>
class Dense{
    static_assert(std::is_arithmetic<T>::value, "Template for Fully connected layer must be of numeric type");
private:

    std::vector<Layer<T>> layers;
public:
    Dense(const std::vector<Layer<T>>& layers){
        if (layers.size() == 0){
            throw std::runtime_error("There must be at least one layer");
        }
        for (unsigned int i = 0; i < layers.size() - 1; i++){
            if (layers[i].out_nodes() != layers[i + 1].in_nodes()){
                throw std::runtime_error("The number of nodes of each layer are mismatch");
            }
        }
        this->layers = layers;
    }

    Dense(const std::vector<Layer<T>>& layers, Utils::Randomizer<T> randFunc) : Dense(layers){
        for (Layer<T>& layer : this->layers){
            for (unsigned int i = 0; i < layer.out_nodes(); i++){
                for (unsigned int j = 0; j < layer.in_nodes(); j++){
                    layer.weight({j, i}) = randFunc.generate();
                }
                layer.bias({0, i}) = randFunc.generate();
            }
        }
    }

    /**
     * Feed forward method. Overloading the () operator
     * 
     * Instead of implementing a method called forward(), I prefer instanceOfDense(input) syntax.
     */
    void operator()(const Matrix<T>& input){
        if (input.getDim()[0] != 1 || input.getDim()[1] != this->layers[0].in_nodes()){
            throw std::runtime_error("The size of the input matrix is mismatch with the Dense layer");
        }

        for (unsigned int i = 0; i < this->layers.size(); i++){

            // Compute the value for each node
            this->layers[i].output = this->layers[i].weight * ((i == 0) ? input : this->layers[i - 1].output) + this->layers[i].bias;

            // Apply activiation function if it is provided.
            if (this->layers[i].actFunc != nullptr){
                for (unsigned int j = 0; j < this->layers[i].out_nodes(); j++){
                    this->layers[i].output({0, j}) = this->layers[i].actFunc(this->layers[i].output({0, j}));
                }
            }
        }
    }

    /**
     * Get a layer of the neural network
     *
     * This method will return any layer of the neural network, the index must start at 1 because
     * class Dense does not keep track of the input layer.
     * 
     */
    Matrix<T> getLayer(unsigned int index){
        if (index < 1 || index > this->layers.size()){
            throw std::runtime_error("Index is out of the number of layer this Dense instance has.");
        }
        return this->layers[index - 1].output;
    }

    // Return the number of layer this Dense instance has, including the input layer.
    unsigned int getNumberOfLayers(){
        return this->layers.size() + 1;
    }

    // Get the string representing the number of nodes of each layer
    std::string toString(){
        std::string result = std::to_string(this->layers[0].in_nodes()) + " -> ";
        for (unsigned int i = 0; i < this->layers.size(); i++){
            result += (std::to_string(this->layers[i].out_nodes()) + ((i == this->layers.size() - 1) ? "" : " -> "));
        }
        return result;
    }
};




#endif