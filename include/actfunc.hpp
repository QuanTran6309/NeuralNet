#ifndef ACTFUNC
#define ACTFUNC

#include <type_traits>
#include <cmath>

namespace ActFunc{

template<typename T>
T sigmoid(T x){
    static_assert(std::is_arithmetic<T>::value, "Numeric value only");
    return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
};

template<typename T>
T Relu(T x){
    static_assert(std::is_arithmetic<T>::value, "Numeric value only");
    return (x > 0) ? x : 0;
};

}

#endif

