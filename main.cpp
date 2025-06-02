
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
    
    /*
    Matrix<unsigned int> image = mnist.getImage(0);
    image.reShape({1, mnist.getTotalPixelsPerImage()});
    model.forward(image);
    */

    Tensor<unsigned int> tensor({3, 2, 4});

    unsigned int src_tensor[24] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24};
    tensor.setTensor(src_tensor, 24);

    std::cout << tensor.toString() << std::endl;
    return 0;
}

