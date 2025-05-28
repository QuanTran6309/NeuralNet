
#ifndef TENSOR
#define TENSOR
#include <stdexcept>
#include <random>  
#include <type_traits>
#include <memory>
#include <vector>

#include "utils.hpp"



struct Range {
    unsigned int start; // inlcusive
    unsigned int end;   // inclusive
};

template<typename T>
class Tensor {
    static_assert(std::is_arithmetic<T>::value, "Tensor only takes numeric values");

private:
    std::shared_ptr<T[]> tensor;
    std::vector<unsigned int> dimensions;

protected:
    /**
     * Get a direct access modifier of the tensor.
     * 
     * Must be careful with this because any changes to the returned pointer 
     * can affect the original value of the tensor.
     */
    std::shared_ptr<T[]> getTensorPtr(){
        return this->tensor;
    }

    // Translate the position vector to the offset on the memory
    unsigned int posVecToIndex(const std::vector<unsigned int>& pos){
        if (pos.size() > this->dimensions.size() || pos.size() == 0){
            throw std::runtime_error("Position vector is invalid");
        }
        unsigned int prevSize = 1;
        unsigned int index = 0;
        for (unsigned int i = 0; i < pos.size(); i++){
            index += (pos[i] * prevSize);
            prevSize *= this->dimensions[i];
        }

        if (index >= this->getTotalEntries()){
            throw std::runtime_error("Index out of bound");
        }

        return index;
    }
public:
    // Virtual destructor for polymorphism
    virtual ~Tensor() = default;
    Tensor() = default;

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
        this->tensor = std::shared_ptr<T[]>(new T[this->getTotalEntries()], std::default_delete<T[]>());
    }


    // Construct a tensor from another tensor (vector form)
    Tensor (std::vector<T> src_tensor, std::vector<unsigned int> dimensions){
        this->tensor = std::shared_ptr<T[]>(new T[src_tensor.size()], std::default_delete<T[]>());
        for (unsigned int i = 0; i < src_tensor.size(); i++){
            this->tensor[i] = src_tensor[i];
        }
        this->dimensions = dimensions;
    }

    
    // Construct a tensor from another tensor (array form)
    Tensor (T *src_tensor, std::vector<unsigned int> dimensions){
        this->dimensions = dimensions;
        unsigned int total_entries = this->getTotalEntries();

        this->tensor = std::shared_ptr<T[]>(new T[total_entries], std::default_delete<T[]>());
        for (unsigned int i = 0; i < total_entries; i++){
            this->tensor[i] = src_tensor[i];
        }
    }

    // Honestly I just know how to implement tensor plus, minus
    // I have not really found a way to implement tensor cross product.
    Tensor<T> operator+(const Tensor<T>& other) const {
        if (this->getDim() != other.getDim()){
            throw std::runtime_error("Cannot perform addition on two different tensors");
        }

        // Get the access to the other tensor
        const T *otherTensorPtr = other.begin();

        // Create new tensor instance
        Tensor<T> newTensor(this->getDim());
        std::shared_ptr<T[]> newTensorPtr = newTensor.getTensorPtr();
        
        // Perform addition on this new tensor
        for (unsigned int i = 0; i < other.getTotalEntries(); i++){
            newTensorPtr[i] = this->tensor[i] + otherTensorPtr[i];
        }

        return newTensor;
    }

    Tensor<T> operator-(const Tensor<T>& other) const {
        if (this->getDim() != other.getDim()){
            throw std::runtime_error("Cannot perform addition on two different tensors");
        }

        // Get the access to the other tensor
        const T *otherTensorPtr = other.begin();

        // Create new tensor instance
        Tensor<T> newTensor(this->getDim());
        std::shared_ptr<T[]> newTensorPtr = newTensor.getTensorPtr();
        
        // Perform subtraction on this new tensor
        for (unsigned int i = 0; i < other.getTotalEntries(); i++){
            newTensorPtr[i] = this->tensor[i] - otherTensorPtr[i];
        }

        return newTensor;
    }

    Tensor<T> operator-() const {
        // Create new tensor instance
        Tensor<T> newTensor(this->getDim());
        std::shared_ptr<T[]> newTensorPtr = newTensor.getTensorPtr();
        
        // Perform invertion on this new tensor
        for (unsigned int i = 0; i < newTensor.getTotalEntries(); i++){
            newTensorPtr[i] = -this->tensor[i];
        }

        return newTensor;
    }



    // Get the total number of entries of this tensor, depending on the dimension size
    size_t getTotalEntries() const {
        return std::accumulate(this->dimensions.begin(), this->dimensions.end(), 1, std::multiplies<unsigned int>());
    }

    // Iterator pattern : not really as safe as a true iterator.
    const T* begin() const{
        return this->tensor.get();
    }
    const T* end() const{
        return this->tensor.get() + this->getTotalEntries();
    }
    
    // Get the size of the tensor
    std::vector<unsigned int> getDim() const{
        return this->dimensions;
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

        for (size_t i = 0; i < total_entries; i++){
            this->tensor[i] = Utils::getRealRandom(-100, 100);
        }
    }

    /**
     * Set the entire tensor's entries from src_tensor
     * 
     * The number of entries of src_tensor must match with this tensor.
     */
    void setTensor(T *src_tensor, size_t num_entries){
        if (num_entries != this->getTotalEntries()){
            throw std::runtime_error("src_tensor has different number of entries than this tensor");
        }

        // If the tensor is sharing the memory with
        if (this->tensor.use_count() > 1){
            this->tensor = std::shared_ptr<T[]>(new T[num_entries], std::default_delete<T[]>());
        }
        std::copy(src_tensor, src_tensor + num_entries, this->tensor.get());
    }

    /**
     * Set value for a single entry using a single index 
     * 
     * This index must be the offset on the platten tensor
     */
    void setEntry(unsigned int index, T value){
        if (index >= this->getTotalEntries()){
            throw std::runtime_error("Index out of bound");
        }
        this->tensor[index] = value;
    }

    /**
     * Set value for a single entry using vector
     * 
     * The vector must reflect the position of the entry.
     * The number of elements of the position vector cannot exceed the number
     * of elements of the tensor's dimension.
     */
    void setEntry(const std::vector<unsigned int>& pos, T value){
        unsigned int index = this->posVecToIndex(pos);
        this->tensor[index] = value;
    }

    // entry is vector indicating the position of the entry in the tensor user want to get
    // The order of value in entry is from the least to most significant dimension.
    T getEntry(const std::vector<unsigned int>& pos) const{
        return this->tensor[this->posVecToIndex(pos)];
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

        // To iterate a tensor in one dimension, we have to use [index * (product of all previous dimension size)]
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
                indices = Utils::sumOfCartesianProd({indices, index});
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


#endif
