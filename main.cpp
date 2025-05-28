#include "linear.hpp"
#include "actfunc.hpp"
#include "data/mnist.hpp"
#include "matrix.hpp"
#include <iostream>




int main(void){
    DataEngine::MNIST mnist("MNIST/train/images");

    std::cout << mnist.getTotalPixelsPerImage() << std::endl;

    Linear<float> inputLayer(mnist.getTotalPixelsPerImage(), 500, ActFunc::Relu);
    Linear<float> hiddenLayer(500, 400, ActFunc::Relu);
    Linear<float> outLayer(500, 400, ActFunc::Relu);

    

    return 0;
}

