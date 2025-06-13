

#include <cuda_runtime.h>

template<typename T>
__global__
void addTensor(const T *d_tensor1, const T *d_tensor2, T *d_tensor, unsigned int totalEntries){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < totalEntries){
        d_tensor[i] = d_tensor1[i] + d_tensor2[i];
    }
}

