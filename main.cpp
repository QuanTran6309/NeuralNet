#include <iostream>
#include "algebra/tensor.hpp"
#include <string>
#include <vector>


int main(int argc, char **argv){

    
    float matrix1[] = {0, 1, 2, 
                       3, 4, 5, 
                       6, 7, 8,
                       10, 10, 10
                    };
    IdioticML::Tensor tensor1({3, 2, 2}, matrix1, IdioticML::DataType::DataType::FLOAT, true);


    float matrix2[] = {0, 1, 2, 
                       3, 6, 5, 
                       6, 9, 10,
                       10, 10, 10
                    };
    IdioticML::Tensor tensor2({3, 2, 2}, matrix2, IdioticML::DataType::DataType::FLOAT, true);


    IdioticML::Tensor tensor = tensor1 + tensor2;
    

    std::cout << tensor.toString() << std::endl;

}