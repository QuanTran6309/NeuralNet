
#ifndef MATRIX
#define MATRIX


#include "tensor.hpp"

template<typename T>
class Matrix : public Tensor<T>{
    static_assert(std::is_arithmetic<T>::value, "Tensor only takes numeric values");
    
public:
    Matrix() = default;

    // Init an empty matrix
    Matrix(unsigned int columns, unsigned int rows) : Tensor<T>({columns, rows}){}
    
    // Init a matrix from nested vector
    Matrix(const std::vector<std::vector<T>>& src_mat) : Tensor<T>({src_mat[0].size(), src_mat.size()}){
        for (unsigned int i = 1; i < src_mat.size(); i++){
            if (src_mat[i].size() != src_mat[0].size()){
                throw std::runtime_error("Matrix size is not legit");
            }
        }
        
        // Copy value into the Matrix
        std::shared_ptr<T[]> matrixPtr = this->getTensorPtr();
        for (unsigned int i = 0; i < src_mat.size(); i++){
            for (unsigned int j = 0; j < src_mat[0].size(); j++){
                matrixPtr[i * src_mat[0].size() + j] = src_mat[i][j];
            }
        }
    }

    Matrix(T *src_tensor, const std::vector<unsigned int>& dimensions) : Tensor<T>(src_tensor, dimensions){
        if (dimensions.size() > 2){
            throw std::runtime_error("Matrix is not legit");
        }
    }

    // Init a matrix from given Tensor
    Matrix(const Tensor<T>& tensor) : Tensor<T>(tensor.getDim()){
        if (tensor.getDim().size() > 2){
            throw std::runtime_error("Matrix size is invalue");
        }
        std::copy(tensor.begin(), tensor.end(), this->getTensorPtr().get());
    }

    // Compatible with different primitive data type
    template<typename U>
    Matrix(const Matrix<U>& other) : Tensor<T>(other.getDim()){
        if (other.getDim().size() > 2){
            throw std::runtime_error("Matrix size is invalid");
        }

        for (unsigned int i = 0; i < this->totalEntries; i++){
            this->tensor[i] = static_cast<T> (*(other.begin() + i));
        }
    }


    // CPU matrix cross product.
    Matrix<T> operator*(const Matrix<T>& other) const{
        // Check if the size of two matrices fit
        std::vector<unsigned int> thisDim = this->getDim();
        std::vector<unsigned int> otherDim = other.getDim();
        if (thisDim[0] != otherDim[1]){
            throw std::runtime_error("Matrices size is not fit for cross product");
        }
        
        // Get the matrix pointer
        const T *otherPtr = other.begin();
        const T *thisPtr = this->begin();

        // Create maxtrix instance
        Matrix<T> newMatrix(otherDim[0], thisDim[1]);
        std::shared_ptr<T[]> newMatrixPtr = newMatrix.getTensorPtr();
        
        // Perform maxtrix cross product
        // Inefficient O(n^3)
        for (unsigned int row = 0; row < thisDim[1]; row++){
            for (unsigned int col = 0; col < otherDim[0]; col++){
                T buffer = 0;
                for (unsigned int itr = 0; itr < thisDim[0]; itr++){
                    buffer += (thisPtr[row * thisDim[0] + itr] * otherPtr[itr * otherDim[0] + col]);
                }

                newMatrixPtr[row * otherDim[0] + col] = buffer;
            }
        }
        return newMatrix;
    }

    Matrix<T> operator+(const Matrix<T>& other) const {
        Tensor<T> result = Tensor<T>::operator+(other);
        return Matrix<T>(result);
    }

    Matrix<T> operator-(const Matrix<T>& other) const {
        Tensor<T> result = Tensor<T>::operator-(other);
        return Matrix<T>(result);
    }

    Matrix<T> operator-() const {
        Tensor<T> newMatrix(this->getDim());

        std::shared_ptr<T[]> newMatrixPtr = newMatrix.getTensorPtr();

        // Get this matrix's entries and copy to buffer
        const T* thisPtr = this->begin();
        for (unsigned int i = 0; i < newMatrix.getTotalEntries(); i++){
            newMatrixPtr[i] = thisPtr[i];
        }

        return Matrix<T>(newMatrix);
    }

    
};


#endif

