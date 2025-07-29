#include <iostream>
#include "algebra/tensor.hpp"
#include <string>
#include <vector>

#include <utility>

using namespace IdioticML;



int main(int argc, char **argv){

    
    float matrix1Arr[] = {89, 1, 2,
                          3, 4, 5,
                          6, 7, 8,
                          0, 0, 0
                    };
    float matrix2Arr[] = {89, 1, 2, 5, 6,
                          3, 4, 5, 90, 9,
                          6, 7, 8, 9, 2
                    };
    
    Tensor matrix1({3, 4}, matrix1Arr, TensorType::FLOAT, Device(DeviceType::CPU));
    Tensor matrix2({5, 3}, matrix2Arr, TensorType::FLOAT, Device(DeviceType::CPU));
    
    Tensor matrix3 = matrix1 * matrix2;

    Tensor slicedMatrix = matrix3.slice({
        {4, 0},
        {2, 0}
    });


    std::cout << matrix3.toString() << std::endl;

    std::cout << "Sliced tensor" << std::endl;
    std::cout << slicedMatrix.toString() << std::endl;

    return 0;
}