
#include "algebra/tensor.cuh"
#include <iostream>
    
using namespace IdioticML;
int main(){
    
    /*
    const int N = 1000;
    float A[N], B[N], C[N];

    for (int i = 0; i < N; ++i) {
        A[i] = i;
        B[i] = 2 * i;
    }
    vectorAdd(A, B, C, N);
    std::cout << "C[0] = " << C[0] << ", C[999] = " << C[999] << std::endl;
    */

    int src1[] = {1, 2, 3, 4, 5, 6};
    int src2[] = {3, 4, 5, 6, 7, 8};
    Tensor tensor1({6, 1}, INT, src1, true);
    Tensor tensor2({6, 1}, INT, src2, true);

    Tensor tensor3 = tensor1 + tensor2;
    tensor3.printTensor();

    return 0;
}
