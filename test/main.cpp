#include <iostream>
#include "algebra/tensor.hpp"
#include <string>
#include <vector>

using namespace IdioticML;


class LinAlg {
public:
    LinAlg(){
        std::cout << "Constructor" << std::endl;
    }
    ~LinAlg(){
        std::cout << "Destructor" << std::endl;
    }

    LinAlg add(){
        return LinAlg();
    }
};

int main(int argc, char **argv){

    
    float matrixArr[] = {89, 1, 2,
                        3, 4, 5,
                        6, 7, 8
                    };
    Tensor matrix1({3, 3}, matrixArr, TensorType::FLOAT, Device(DeviceType::GPU));
    Tensor matrix2({3, 3}, matrixArr, TensorType::FLOAT, Device(DeviceType::GPU));
    

    Tensor matrix3 = matrix1 + matrix2;

    std::cout << matrix3.toString() << std::endl;

    return 0;
}