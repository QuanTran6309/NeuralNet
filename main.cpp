
#include "actfunc.hpp"
#include "matrix.hpp"
#include "data/mnist.hpp"
#include "layer.hpp"
#include "dense.hpp"
#include <iostream>


int main(void){
    DataEngine::MNIST mnist("MNIST/train/images");

    Dense<float> model({
        {784, 600, ActFunc::Relu},
        {600, 600, ActFunc::Relu},
    });




    return 0;
}

