
#include "actfunc.hpp"
#include "matrix.hpp"
#include "data/mnist.hpp"
#include "layer.hpp"
#include "dense.hpp"
#include <iostream>
#include "utils.hpp"


int main(void){
    DataEngine::MNIST mnist("MNIST/train/images");

    Matrix<float> image = mnist.getImage(0);
    image.reShape({1, 784});
    Dense<float> model(
        {
            Layer<float>(784, 60, ActFunc::Relu),
            Layer<float>(60, 10, ActFunc::Relu),
            Layer<float>(10, 5, ActFunc::Relu)
        }
       // Utils::Randomizer<float>(Utils::getRealRandom, -100, 100)
    );

    model(image);

    std::cout << model.getLayer(1).toString() << std::endl;


    return 0;
}

