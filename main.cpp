#include "linear.hpp"
#include "actfunc.hpp"
#include <iostream>




int main(void){
    Linear<float> input(256, 32, ActFunc::sigmoid);
    
    return 0;
}

