#include <iostream>
#include <string>
#include <vector>
#include <utility>

#include "algebra/tensor.hpp"

using namespace IdioticML;


int main(int argc, char **argv){

    
    float matrix1Arr[] = {89, 1, 2,
                          3, 4, 5,
                          6, 7, 8
                    };
    float matrix2Arr[] = {89, 1, 2, 5, 6,
                          3, 4, 5, 90, 9,
                          6, 7, 8, 9, 2
                    };
    
    Tensor matrix1({3, 3}, matrix1Arr, TensorType::FLOAT, Device(DeviceType::CPU));
    Tensor matrix2({5, 3}, matrix2Arr, TensorType::FLOAT, Device(DeviceType::CPU));
    
    Tensor matrix3 = matrix1 * matrix2;

    std::cout << matrix3.toString() << std::endl;

    return 0;
}