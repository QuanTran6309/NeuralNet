#include <iostream>
#include "algebra/tensor.hpp"
#include <string>
#include <vector>

using namespace IdioticML;

int main(int argc, char **argv){

    
    float matrix1[] = {89, 1, 2,
                        3, 4, 5,
                        6, 7, 8
                    };
    Tensor matrix1({3, 3}, matrix1, TensorType::FLOAT, Device(DeviceType::GPU));

    Tensor matrix2({3, 3}, matrix1, TensorType::FLOAT, Device(DeviceType::GPU));


    matrix1 = instanceOfLinAlg.add()

    
    

    return 0;
}