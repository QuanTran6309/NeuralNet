
#ifndef TENSOR
#define TENSOR

#include <stdexcept>
#include <vector>

struct Range {
    unsigned int start; // inlcusive
    unsigned int end;   // inclusive
};

enum TensorType {
    FLOAT,
    DOUBLE,
    INT,
    UINT
};

class Tensor {
private:

protected:
    std::vector<unsigned int> dimensions;
    // This can be on either HOST or DEVICE (GPU) depending on the parameters passed to constructor
    void *tensor; 
    unsigned int totalEntries;
    TensorType type;
    bool is_onCUDA = false;

public:
    
    // Virtual destructor for polymorphism
    virtual ~Tensor();
    Tensor() = default;

    // Just set the dimension and size without assigning any entries
    Tensor(const std::vector<unsigned int>& dimensions, 
           TensorType type,
           bool to_cuda = false);

    // Set the tensor's entries from src_tensor and also set the dimensions
    Tensor (const void *src_tensor, 
            const std::vector<unsigned int>& dimensions, 
            TensorType type,
            bool to_cuda = false);

    void printTensor();

    // Overload operator +
    Tensor operator+(const Tensor& other) const;
};



void vectorAdd(const float* A, const float* B, float* C, int N);



#endif
