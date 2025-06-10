
#include "tensor.hpp"
#include <iostream>
    
int main(){
    float arr[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        Tensor<float> tensor(arr, {3, 1, 3});
        
        // Let say you want to access the entry at second column, first row, and third channel.
        std::cout << tensor({1, 0, 2}) << std::endl;
        

        return 0;
}
