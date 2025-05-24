#include "layer.cc"
#include "weights.cc"
#include <iostream>


class Model {
private:
    std::vector<Layer<float>> layers;
    std::vector<Weights<float>> weights;
public:
    Model(Tensor<float> operand){
        
            std::cout << (int)operand.getType() << std::endl;
        
    }

};


int main(void){
    Layer<float> tensor = Layer<float>(34);
    Model nn(tensor);
    
    //std::cout << (int)tensor.getType() << std::endl;
    return 0;
}

