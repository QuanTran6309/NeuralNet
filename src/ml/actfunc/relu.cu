#include "ml/actfunc/relu.hpp"
namespace IdioticML{


inline void ReluCPU(){
    
}

inline void ReluGPU(){

}

void Relu::compute(Tensor& tensor){
    if (tensor.getDeviceType() == DeviceType::CPU){
        ReluCPU();    
    }
    else {
        ReluGPU();
    }
}




}