#include <iostream>
#include <string>
#include <vector>
#include <utility>



enum DTypes {
    FLOAT,
    DOUBLE
};

// Type trait mapping DType to actual C++ type
template <DTypes> struct DTypeToType;

template <> struct DTypeToType<DTypes::FLOAT> { using type = float; };
template <> struct DTypeToType<DTypes::DOUBLE> { using type = double; };


template <typename T>
void printOut(const T* data){
    std::cout << data[0] << std::endl;
}

int main(int argc, char **argv){

    
    return 0;
}