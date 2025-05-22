#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <random>  
#include <type_traits>
#include <memory>

// Produce the cartesian product and sum each element within each set
std::vector<unsigned int> sumOfCartesianProd(std::vector<std::vector<unsigned int>> sets){
    if (sets.size() == 1){
        return sets[0];
    }
    std::vector<unsigned int> result;
    std::vector<std::vector<unsigned int>> subSet(sets.begin() + 1, sets.end());
    std::vector<unsigned int> deeperSet = sumOfCartesianProd(subSet);
    for (unsigned int curr : sets[0]){
        for (unsigned int deeper : deeperSet){
            result.push_back(curr + deeper);
        }
    }
    return result;
}

struct Range {
    unsigned int start; // inlcusive
    unsigned int end;   // inclusive
};


template<typename T>
class Tensor {
    static_assert(std::is_arithmetic<T>::value, "Tensor only takes numeric values");

private:
    std::unique_ptr<T[]> tensor;
    std::vector<unsigned int> dimensions;

public:
    /**
     * Init a Tensor by size, randomly assign each entries
     * 
     * @param dim_size: the size of each dimension, the number at index 0 coresspond to the highest order of dimension.
     * 
     * Example: Tensor({3, 1, 2}) corresponds to a tensor with 3 dimension : 3 columns, 1 row, 2 layers
     */
    Tensor(std::vector<unsigned int> dimensions){
        if (dimensions.size() == 0){
            throw std::runtime_error("The dimensions cannot be empty");
        }
        this->dimensions = dimensions;

        // Set the size for the tensor
        size_t total_entries = 1;
        for (unsigned int dim : dimensions){
            if (dim == 0){
                throw std::runtime_error("Dimension of a tensor cannot be 0");
            }
            total_entries *= dim;
        }
        this->tensor = std::make_unique<T[]>(total_entries);
    }


    // Construct a tensor from another tensor (vector form)
    Tensor (std::vector<T> src_tensor, std::vector<unsigned int> dimensions){
        this->tensor = std::make_unique<T[]>(src_tensor.size());
        for (unsigned int i = 0; i < src_tensor.size(); i++){
            this->tensor[i] = src_tensor[i];
        }

        this->dimensions = dimensions;
    }

    // Construct a tensor from another tensor (array form)
    Tensor (T *src_tensor, std::vector<unsigned int> dimensions){
        unsigned int total_entries = 1;
        for (unsigned int dim : dimensions){
            total_entries *= dim;
        }

        this->tensor = std::make_unique<T[]>(total_entries);
        for (unsigned int i = 0; i < total_entries; i++){
            this->tensor[i] = src_tensor[i];
        }
        this->dimensions = dimensions;
    }


    // Randomly assign value to all entries of the tensor.
    void randomlyInit(){
        // Get the entire flattened size.
        size_t total_entries = 1;
        for (unsigned int dim : dimensions){
            if (dim == 0){
                throw std::runtime_error("Dimension of a tensor cannot be 0");
            }
            total_entries *= dim;
        }
        
        // Randomly assign the entries.
        std::random_device rd;  
        std::mt19937 gen(rd()); 
        std::uniform_real_distribution<T> dist(-100, 100);
        for (size_t i = 0; i < total_entries; i++){
            this->tensor[i] = dist(gen);
        }
    }

    // entry is vector indicating the position of the entry in the tensor user want to get
    // The order of value in entry is from the least to most significant dimension.
    T getEntry(std::vector<unsigned int> entry){
        if (entry.size() > this->dimensions.size()){
            throw std::runtime_error("The entry has more dimensions than the matrix");
        }

        if (entry[0] >= this->dimensions[0]){
            throw std::runtime_error("Entry does not exist");
        }

        // Calculate the index basde on the entry vector, dim0 + dim1 * size0 + dim2 * size0 * size1, and so on
        unsigned int index = entry[0];
        unsigned int prev_dimSize = 1;
        for (int i = 1; i < entry.size(); i++){
            if (entry[i] >= this->dimensions[i]){
                throw std::runtime_error("Entry does not exist");
            }
            prev_dimSize *= this->dimensions[i];
            index += (entry[i] * prev_dimSize);
        }

        return this->tensor[index];
    }


    // Get the size of the tensor
    std::vector<unsigned int> getDim(){
        return this->dimensions;
    }

    // Get just a portion of a tensor,
    // Tensor slicing
    Tensor<T> getSlicedTensor(std::vector<Range> bounds){
        if (bounds.size() > this->dimensions.size()){
            throw std::runtime_error("Bounds' dimension exceeds tensor's dimension");
        }

        // Used to store the indices for each entry that we'll copy from source tensor
        std::vector<unsigned int> indices;
        
        // Get the new dimension for the new tensor
        std::vector<unsigned int> newDim;

        // It's going backward so also need backward prevSize
        unsigned int prevSize = 1;
        for (unsigned int i = 0; i < this->dimensions.size() - 1; i++){
            prevSize *= this->dimensions[i];
        }
        
        /**
         * For example: bounds = {
         *                        {1:3}
         *                        {0:2}
         *                        {2:4} 
         *                       }
         * The loop will convert that to
         *              {
         *              {1, 2, 3},
         *              {0, 1, 2} * 0th dim,
         *              {2, 3, 4} * 0th dim * 1th dim
         *              }
         * and finally perform cartesian product and sum all of elements within each set to get the set
         * of indices to copy from
         *
         * We need to iterate from the bottom of the bounds to ensure the ascending order of the offsets
         */
        for (int i = bounds.size() - 1; i >= 0; i--){
            Range bound = bounds[i];
            std::vector<unsigned int> index;

            // 1:4 get all entries from 1 to 4
            if (bound.start <= bound.end){
                if (bound.end >= this->dimensions[i]){
                    throw std::runtime_error("Index out of bound");
                }
                unsigned int dimSize = bound.end - bound.start + 1;

                // Insert the size for this dimension to newDim
                newDim.insert(newDim.begin(), dimSize);
                
                // Insert indices 
                for (unsigned int i = bound.start; i <= bound.end; i++){
                    index.push_back(i * prevSize);
                }
            }
            // 4:1 get all entries from 0->1 and from 4->end
            else {
                if (bound.start >= this->dimensions[i]){
                    throw std::runtime_error("Index out of bound");
                }
                unsigned int dimSize = bound.end + 1 + (this->dimensions[i] - bound.start);

                // Insert the size for this dimension to newDim
                newDim.insert(newDim.begin(), dimSize);

                // Insert indices 
                for (unsigned int j = 0; j <= bound.end; j++){
                    index.push_back(j * prevSize);
                }
                for (unsigned int j = bound.start; j < this->dimensions[i]; j++){
                    index.push_back(j * prevSize);
                }
            }
            
            if (i > 0){
                prevSize /= this->dimensions[i - 1];
            }
            
            // Process the indices
            if (indices.size() != 0){
                indices = sumOfCartesianProd({indices, index});
            }
            else{
                indices = index;
            }
        }

        T *newTensor = new T[std::accumulate(newDim.begin(), newDim.end(), 1, std::multiplies<unsigned int>())];
        for (unsigned int i = 0; i < indices.size(); i++){
            newTensor[i] = this->tensor[indices[i]];
        }
        
        Tensor<T> new_tensor(newTensor, newDim);

        return new_tensor;
    }

};

template<typename T>
class Layer : public Tensor<T>{
private:
    
public:
    // Construct by size, randomly assign values
    Layer(unsigned int size) : Tensor<T>({1, size}){}

    /**
     * Compute the value for this layer's nodes based on the previous layer and weights
     * 
     * Technically, it's a product of a vector and a matrix
     */
    void compute(Layer<T> prev, Tensor<T> weights){
        std::vector<unsigned int> dim = weights.getDim();
        if (prev.getDim()[1] != dim[0] || dim[1] != this->getDim()[1]){
            throw std::runtime_error("weights and previous layer do not have appropriate size to make a product\n");    
        }

        
        T *thisLayer = this->getTensor();
        T *prevLayer = prev->getTensor();
        T *weightsTensor = weights->getTensor();
        
        // weights x prevLayer
        
        

    }
};





class Model {
private:
    std::vector<float> weights;
    std::vector<float> biases;
    std::vector<Layer<float>> layers;
    float learn_rate = 0.01;
public:


    
};




int main(void){
    
    Tensor<float> tensor({4,3,2});
    tensor.getTensor({
        {2, 3},
        {0, 1},
        {1, 1}
    });
    return 0;
}