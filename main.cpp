#include "linear.hpp"
#include "actfunc.hpp"
#include "data/mnist.hpp"
#include <iostream>




int main(void){
    DataEngine::MNIST mnist("MNIST/train/images");

    std::cout << mnist.getTotalPixelsPerImage() << std::endl;
    Linear<float> inputLayer(mnist.getTotalPixelsPerImage(), , ActFunc::Relu);
    Linear<float> hiddenLayer(90, 10, ActFunc::Relu);
    return 0;
}

