

#include "algebra/tensor.hpp"

using namespace IdioticML;

void Nothing(Tensor<int> tensor){
    Tensor<int> buffer = tensor;
}


int main(){

    int src1[] = {1, 2, 3, 4, 5, 6};
    int src2[] = {3, 4, 5, 6, 7, 8};

    Tensor<int> matrix1({3, 2}, src1);    
    Tensor<int> matrix2({3, 2}, src2);
    
    Tensor<int> tensor = matrix1 + matrix2;

    std::cout << tensor.toString() << std::endl;
    
    Nothing(tensor);

    std::cout << tensor.toString() << std::endl;
    
    return 0;
}
