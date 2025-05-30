
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
        for (unsigned int i = 0; i < layers.size(); i++){
            layers[i].compute(layers[i - 1].)
        }
    }

    
}

int main(void){
    DataEngine::MNIST mnist("MNIST/train/images");

    std::cout << mnist.getTotalPixelsPerImage() << std::endl;

    Layer inputLayer(mnist.getTotalPixelsPerImage(), 500, ActFunc::Relu);
    Layer hiddenLayer(500, 400, ActFunc::Relu);
    Layer outLayer(400, 10, ActFunc::Relu);

    inputLayer.randomlyInit();
    hiddenLayer.randomlyInit();
    outLayer.randomlyInit();


    //Matrix<unsigned int> image = mnist.getImage(0);
    

    return 0;
}

