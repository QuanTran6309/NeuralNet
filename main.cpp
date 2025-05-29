
#include "layer.hpp"
#include "actfunc.hpp"
#include "matrix.hpp"

#include "data/mnist.hpp"
#include <iostream>




int main(void){
    DataEngine::MNIST mnist("MNIST/train/images");

    std::cout << mnist.getTotalPixelsPerImage() << std::endl;

    Layer inputLayer(mnist.getTotalPixelsPerImage(), 500, ActFunc::Relu);
    Layer hiddenLayer(500, 400, ActFunc::Relu);
    Layer outLayer(400, 10, ActFunc::Relu);

    inputLayer.randomlyInit();
    hiddenLayer.randomlyInit();
    outLayer.randomlyInit();

    Matrix<unsigned int> weights = inputLayer.weights;
    cv::Mat image(weights.getDim()[1], weights.getDim()[0], CV_8UC1, weights.tensor.get());
    cv::imshow("WEIGHTS", image);
    cv::waitKey(0);


    //Matrix<unsigned int> image = mnist.getImage(0);
    

    return 0;
}

