#include <iostream>
#include "algebra/tensor.hpp"
#include "algebra/matrix.hpp"
#include <string>
#include <vector>
#include <cuda_runtime.h>

int main(int argc, char **argv){

    
    float matrix1[] = {0, 1, 2,
                        3, 4, 5,
                        6, 7, 8
                    };
    IdioticML::Matrix mat1(3, 3, matrix1, IdioticML::DataType::DataType::FLOAT, true);


    float matrix2[] = {0, 1, 2,
                        3, 4, 5,
                        6, 7, 9
                    };
    IdioticML::Matrix mat2(3, 3, matrix2, IdioticML::DataType::DataType::FLOAT, true);

    IdioticML::Matrix mat = mat1 + mat2;
    std::cout << mat.toString() << std::endl;

    return 0;
}