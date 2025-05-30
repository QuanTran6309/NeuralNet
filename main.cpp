
#include "layer.hpp"
#include "actfunc.hpp"
#include "matrix.hpp"
#include "data/mnist.hpp"
#include <iostream>


class Model{
private:
    std::vector<Layer> layers;
public:
    Model(const std::vector<Layer>& network, bool is_randomInit = true){
        this->layers = network;

        if (is_randomInit){
            for (Layer layer : this->layers){
                layer.randomlyInit();
            }   
        }
    }

    void forward(Matrix<float> input){
        layers[0].compute(input);
        for (unsigned int i = 1; i < layers.size(); i++){
            layers[i].compute(layers[i - 1].prevOut());
        }
    }

    Matrix<float> outLayer(){
        return this->layers[this->layers.size() - 1].prevOut();
    }
};

int main(void){
    DataEngine::MNIST mnist("MNIST/train/images");


    Model model({
        Layer(mnist.getTotalPixelsPerImage(), 500, ActFunc::Relu),
        Layer(500, 400, ActFunc::Relu),
        Layer(400, 10, ActFunc::Relu)
    });

    model.forward(mnist.getImage(0));

    //Matrix<unsigned int> image = mnist.getImage(0);
    

    return 0;
}

