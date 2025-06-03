
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
    Matrix(const std::vector<std::vector<T>>& src_mat) : Tensor<T>({(unsigned int)src_mat[0].size(), (unsigned int)src_mat.size()}){
        for (unsigned int i = 1; i < src_mat.size(); i++){
            if (src_mat[i].size() != src_mat[0].size()){
                throw std::runtime_error("Matrix size is not legit");
            }
        }

        for (unsigned int i = 0; i < src_mat.size(); i++){
            for (unsigned int j = 0; j < src_mat[0].size(); j++){
                this->tensor[i * src_mat[0].size() + j] = src_mat[i][j];
            }
        }
    }

    // Primarily used for operators overloading.
    Matrix(const T *src_tensor, const std::vector<unsigned int>& dimensions) : Tensor<T>(src_tensor, dimensions){
        if (dimensions.size() > 2){
            throw std::runtime_error("Matrix is not legit");
        }
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
        if (this->dimensions[0] != other.dimensions[1]){
            throw std::runtime_error("Matrices size is not fit for cross product");
        }
        
        // Create maxtrix instance
        Matrix<T> newMatrix(other.dimensions[0], this->dimensions[1]);
        
        // Perform maxtrix cross product
        // Inefficient O(n^3)
        for (unsigned int row = 0; row < this->dimensions[1]; row++){
            for (unsigned int col = 0; col < other.dimensions[0]; col++){
                for (unsigned int itr = 0; itr < this->dimensions[0]; itr++){
                    newMatrix({col, row}) += (this->tensor[row * this->dimensions[0] + itr] * other({col, itr}));
                }
            }
        }
        return newMatrix;
    }

    Matrix<T> operator+(const Matrix<T>& other) const {
        Tensor<T> result = Tensor<T>::operator+(other);
        return Matrix<T>(result.begin(), result.getDim());
    }

    Matrix<T> operator-(const Matrix<T>& other) const {
        Tensor<T> result = Tensor<T>::operator-(other);
        return Matrix<T>(result.begin(), result.getDim());
    }

    // Unary minus operator
    Matrix<T> operator-() const {
        Tensor<T> newMatrix(this->dimensions);

        for (unsigned int i = 0; i < newMatrix.getTotalEntries(); i++){
            newMatrix.tensor[i] = - this->tensor[i];
        }

        return Matrix<T>(newMatrix.begin(), newMatrix.getDim());
    }

    
};


#endif

