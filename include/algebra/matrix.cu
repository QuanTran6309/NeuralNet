#include "matrix.hpp"
#include "vendor/cublas.cuh"
#include "vendor/logger.cuh"

namespace IdioticML{
Matrix::Matrix(unsigned int columns,
               unsigned int rows,
               void *src_tensor,
               DataType::DataType type,
               bool isOnGPU) 
               : Tensor({columns, rows}, src_tensor, type, isOnGPU){}

//
Matrix::Matrix(Tensor& other) : Tensor(other)
{
    if (other.getDim().size() != 2){
        throw std::runtime_error("Matrix size is invalid");
    }
}

// Get the number of rows and columns of the matrix
unsigned int Matrix::rows() const{ return this->dimensions[1]; }
unsigned int Matrix::cols() const{ return this->dimensions[0]; }


/**
 * Overload the * operator for matrix multiplication
 */
inline void CPUmatrixMult_helper(const std::vector<unsigned int>& thisDimensions,
                                 const std::vector<unsigned int>& otherDimensions,
                                 const void* thisTensorPtr,
                                 const void* otherTensorPtr,
                                 void *newTensorPtr,
                                 DataType::DataType type)
{
    switch (type)
    {
    case DataType::DataType::FLOAT:
        for (unsigned int row = 0; row < thisDimensions[1]; row++){
            for (unsigned int col = 0; col < otherDimensions[0]; col++){
                for (unsigned int itr = 0; itr < thisDimensions[0]; itr++){
                    ((float *)newTensorPtr)[col + row * otherDimensions[0]] += ( ((float *)thisTensorPtr)[row * thisDimensions[0] + itr] * ((float *)otherTensorPtr)[col + itr * otherDimensions[0]] );
                }
            }
        }
        return;
    case DataType::DataType::DOUBLE:
        for (unsigned int row = 0; row < thisDimensions[1]; row++){
            for (unsigned int col = 0; col < otherDimensions[0]; col++){
                for (unsigned int itr = 0; itr < thisDimensions[0]; itr++){
                    ((double *)newTensorPtr)[col + row * otherDimensions[0]] += ( ((double *)thisTensorPtr)[row * thisDimensions[0] + itr] * ((double *)otherTensorPtr)[col + itr * otherDimensions[0]] );
                }
            }
        }
        return;
    default:
        break;
    }   
}
inline void GPUmatrixMult_helper(const std::vector<unsigned int>& thisDimensions,
                                 const std::vector<unsigned int>& otherDimensions,
                                 const void* thisTensorPtr,
                                 const void* otherTensorPtr,
                                 void *newTensorPtr,
                                 DataType::DataType type)
{
    switch (type)
    {
    case DataType::DataType::FLOAT:{
        float a = 1;
        float b = 0;
        CUBLAS_CHECK(cublasSgemm(CublasSingleton::getHandler(),
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_N,
                                 otherDimensions[0], thisDimensions[1], thisDimensions[0],
                                 &a,
                                 (float *)otherTensorPtr, otherDimensions[0],
                                 (float *)thisTensorPtr,  thisDimensions[0],
                                 &b,
                                 (float *)newTensorPtr, otherDimensions[0]));   
        break;
    }
    case DataType::DataType::DOUBLE: {
        double a = 1;
        double b = 0;
        CUBLAS_CHECK(cublasDgemm(CublasSingleton::getHandler(),
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_N,
                                 otherDimensions[0], thisDimensions[1], thisDimensions[0],
                                 &a,
                                 (double *)otherTensorPtr, otherDimensions[0],
                                 (double *)thisTensorPtr,  thisDimensions[0],
                                 &b,
                                 (double *)newTensorPtr, otherDimensions[0]));  
        break;
    }
    default:
        break;
    }

    CublasSingleton::releaseHandler();
}
Matrix Matrix::operator*(const Matrix& other) const{
    if (this->dimensions[0] != other.dimensions[1]){
        throw std::runtime_error("Matrices size is not fit for cross product");
    }
    if (this->isOnGPU ^ other.isOnGPU){
        throw std::runtime_error("Both matrices must be on the same device");
    }

    void *newTensorPtr;
    
    if (this->isOnGPU){
        CUDA_CHECK(cudaMalloc(&newTensorPtr ,this->dimensions[1] * other.dimensions[0] * this->entrySize));
        GPUmatrixMult_helper(this->dimensions, other.dimensions, this->tensorPtr, other.tensorPtr, newTensorPtr, this->type);
    }
    else {
        newTensorPtr = std::malloc(this->dimensions[1] * other.dimensions[0] * this->entrySize);
        CPUmatrixMult_helper(this->dimensions, other.dimensions, this->tensorPtr, other.tensorPtr, newTensorPtr, this->type);
    }

    Matrix result(other.dimensions[0], this->dimensions[1], nullptr, this->type, this->isOnGPU);
    result.tensorPtr = (char *)newTensorPtr;

    return result;
}

/**
 * Overload the + operator for the Matrix class
 */
Matrix Matrix::operator+(const Matrix& other) const {
    Tensor result = Tensor::operator+(other);
    return Matrix(result);
}

Matrix Matrix::operator-(const Matrix& other) const {
    Tensor result = Tensor::operator-(other);
    return Matrix(result);
}

}
