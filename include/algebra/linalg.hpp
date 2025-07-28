
#ifndef LINALG
#define LINALG


namespace IdioticML{

/**
 * An abstract class to perform arithmetic operations on Tensor
 * 
 * The idea is that there can be multiple operations at the same time, such as
 *         newTensor = tensor1 + tensor2 + tensor3 * tensor4
 * 
 * Which will be written as
 *         newTensor = LinAlg(tensor1).add(tensor2).add(tensor3).mult(tensor4);
 * 
 * We don't care about the precedence of operators. The users will handle that.
 * 
 * This class is a friend with Tensor. So the constructor of this class has access to the private and protected
 * member of Tensor.
 * 
 * Initially, after the constructor is called:
 *      @param tensorPtr: will be assigned to the same tensorPtr of the given instance of Tensor
 *      @param dimensions: the dimension of the given tensor.
 *      @param isOnGPU: tell the location of the given tensorPtr
 * 
 * Any implementation of this class must keep in mind that:
 *      1) Do not free tensorPtr in the first 
 
class LinAlg{
    std::shared_ptr<DeviceContext> deviceContext;
    char *tensorBufferPtr;
    std::vector<unsigned int> dimensions;
    TensorType type;
    bool isFirstOperation;

    Tensor *destTensor;
public:
    LinAlg() = default;
    LinAlg(Tensor *destTensor, const Tensor& tensor);

    LinAlg& add(const Tensor& other);
    LinAlg& mult(const Tensor& other);
};
*/

class LinAlg {
    static void add();
    static void mult();
};


}



#endif

