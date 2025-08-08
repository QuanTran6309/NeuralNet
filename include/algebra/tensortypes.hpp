#ifndef TENSORTYPES
#define TENSORTYPES

#include <stdexcept>


namespace IdioticML {


enum TensorType {
    FLOAT,
    DOUBLE,
    INT
};

inline unsigned int getTensorTypeSize(TensorType type){
    switch (type)
    {
    case TensorType::FLOAT:
        return sizeof(float);    
    case TensorType::DOUBLE:
        return sizeof(double);
    case TensorType::INT:
        return sizeof(int);
    default:
        throw std::runtime_error("Unsupported data type");
    }
}

/*
template <typename TFunctor>
void dispatch_operation(TensorType type, TFunctor&& functor) {
    
    // The one and only switch statement
    switch (type) {
        case TensorType::FLOAT:
            std::forward<TFunctor>(functor).template operator()<float>((float) 0);
            break;
        case TensorType::DOUBLE:
            std::forward<TFunctor>(functor).template operator()<double>((double) 0);
            break;
        default:
            throw std::runtime_error("Unsupported TensorType");
    }
} */

template <typename Functor>
inline void type_dispatcher(TensorType type, Functor&& functor){
    switch (type)
    {
    case TensorType::FLOAT:
        functor(static_cast<float>(0));
        break;
    case TensorType::DOUBLE:
        functor(static_cast<double>(0));
        break;
    default:
        throw std::runtime_error("Unsupported TensorType");
    }
}

enum DeviceType { GPU, CPU };


}

#endif