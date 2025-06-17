

#include "algebra/tensor.hpp"
#include "algebra/matrix.hpp"

using namespace IdioticML;

void Nothing(Tensor<int> tensor){
    Tensor<int> buffer = tensor;
}


int main(){

    int src1[] = {1, 2, 3, 4, 5, 6};
    int src2[] = {3, 4, 5, 6, 7, 8};

    Matrix<int> matrix1({3, 2}, src1);    
    Matrix<int> matrix2({2, 3}, src2);
    
    Matrix<int> tensor = matrix1 * matrix2;
    
    std::cout << tensor.toString() << std::endl;
    
    return 0;
}
