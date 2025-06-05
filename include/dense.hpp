#ifndef DENSE
#define DENSE

#include "layer.hpp"
#include "matrix.hpp"
#include "utils.hpp"
#include <sstream>


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
            if (layers[i].out_nodes != layers[i + 1].in_nodes){
                throw std::runtime_error("The number of nodes of each layer are mismatch");
            }
        }

        this->layers = layers;
    }


};




#endif