#include "tensor.hpp"


template<typename T>
class Matrix : public Tensor<T>{
    static_assert(std::is_arithmetic<T>::value, "Tensor only takes numeric values");
    
public:
    Matrix() = default;
    Matrix(unsigned int columns, unsigned int rows) : Tensor<T>({columns, rows}){}

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
        Matrix<T> newMatrix(otherDim[0], thisgetDim[1]);
        std::shared_ptr<T[]> newMatrixPtr = newMatrix.getTensorPtr();
        
        // Perform maxtrix cross product
        // Inefficient O(n^3)
        for (unsigned int row = 0; row < thisDim[1]; row++){
            for (unsigned int col = 0; col < otherDim[0]; col++){
                T buffer = 0;
                for (unsigned int itr = 0; itr < thisDim[0]; itr++){
                    buffer += (thisPtr[row * thisDim[0] + itr] * otherPtr[itr * thisDim[0] + col]);
                }

                newMatrixPtr[row * otherDim[0] + col] = buffer;
            }
        }
        return newMatrix;
    }
};

