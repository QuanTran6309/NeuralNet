
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
            for (unsigned int i = 0; i < this->layers.size(); i++){
                this->layers[i].randomlyInit();
            }
        }
    }

    void forward(Matrix<float> input){
        layers[0].compute(input);
        for (unsigned int i = 1; i < layers.size(); i++){
            layers[i].compute(layers[i - 1].prevOut());
        }
    }
    Matrix<float> getLayer(unsigned int i){
        return this->layers[i].getBiases();
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
    
    Matrix<uint8_t> image = mnist.getImage(0);
    image.reShape({1, mnist.getTotalPixelsPerImage()});
    model.forward(image);
    

    std::cout << model.outLayer().toString() << std::endl;

    

    return 0;
}

