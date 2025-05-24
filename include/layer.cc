#ifndef LAYER
#define LAYER

#include <vector>
#include "tensor.cc"



template<typename T>
class Layer : public Tensor<T>{
private:
    
public:
    TensorType getType() override {
        return TensorType::Layer;
    }

    /**
     * Construct by size = the number of nodes, all nodes are not initialized
     * 
     * The dimension of layer is {1, size}, actually it can be just {size} but I want to
     * keep the consistency with the MAXTRIX * VECTOR formula I saw in my linear algebra class.
     * So I order the Layer as 2 dimensional with the first dimension just have size of 1.
     * Technically it's a 1 dimensional array but stands vertically.
     */
    Layer(unsigned int size) : Tensor<T>({1, size}){}
};

#endif


