
#ifndef TENSOR
#define TENSOR

#include <stdexcept>
#include <random>  
#include <type_traits>
#include <memory>
#include <vector>
#include <deque>

#include "utils.hpp"



struct Range {
    unsigned int start; // inlcusive
    unsigned int end;   // inclusive
};

template<typename T>
class Tensor {
    static_assert(std::is_arithmetic<T>::value, "Tensor only takes numeric values");

private:
    /**
     * The actual toString method. This method form the string of the 
     * tensor to be printed on the console.
     * 
     * Magically somehow I implemented to run in linear time. Did not expect this.
     * 
     * Use recursion.
     */
    std::string toStringHelper(std::vector<unsigned int>::iterator itr, 
                               unsigned int offset){
        std::string buffer = "";
        if (itr == this->dimensions.begin()){
            buffer = std::to_string(this->tensor[offset]);
            for (unsigned int i = 1; i < *itr; i++){
                buffer += (", " + std::to_string(static_cast<T>(this->tensor[offset + i])));
            }
            return buffer;
        }

        std::vector<std::string> brac_buffer(*itr);
        for (unsigned int i = 0; i < brac_buffer.size(); i++){
            // What the fuck?
            brac_buffer[i] = "[" + toStringHelper(itr - 1, i * std::accumulate(this->dimensions.begin(), itr, 1, std::multiplies<unsigned int>()) + offset) + "]";
        }

        buffer = brac_buffer[0];
        for (unsigned int i = 1; i < brac_buffer.size(); i++){
            buffer += (",\n" + brac_buffer[i]);
        }

        return buffer;
    }
    
protected:
    

    // Translate the position vector to the offset on the memory
    unsigned int posVecToIndex(const std::vector<unsigned int>& pos) const {
        if (pos.size() > this->dimensions.size() || pos.size() == 0){
            throw std::runtime_error("Position vector is invalid");
        }
        unsigned int prevSize = 1;
        unsigned int index = 0;
        for (unsigned int i = 0; i < pos.size(); i++){
            index += (pos[i] * prevSize);
            prevSize *= this->dimensions[i];
        }

        if (index >= this->totalEntries){
            throw std::runtime_error("Index out of bound");
        }

        return index;
    }
public:

    std::vector<unsigned int> dimensions;
    std::shared_ptr<T[]> tensor;
    unsigned int totalEntries;
    
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
        this->totalEntries = std::accumulate(this->dimensions.begin(), this->dimensions.end(), 1, std::multiplies<unsigned int>());

        // Set the size for the tensor
        this->tensor = std::shared_ptr<T[]>(new T[this->totalEntries], std::default_delete<T[]>());
    }


    // Construct a tensor from another tensor (vector form)
    Tensor (std::vector<T> src_tensor, const std::vector<unsigned int>& dimensions){
        this->totalEntries = std::accumulate(this->dimensions.begin(), this->dimensions.end(), 1, std::multiplies<unsigned int>());
        if (src_tensor.size() != this->totalEntries){
            throw std::runtime_error("The dimension and src_tensor are mismatch");
        }

        this->tensor = std::shared_ptr<T[]>(new T[src_tensor.size()], std::default_delete<T[]>());
        for (unsigned int i = 0; i < src_tensor.size(); i++){
            this->tensor[i] = src_tensor[i];
        }
        this->dimensions = dimensions;
    }

    
    // Construct a tensor from another tensor (array form)
    Tensor (const T *src_tensor, const std::vector<unsigned int>& dimensions){
        this->dimensions = dimensions;
        this->totalEntries = std::accumulate(this->dimensions.begin(), this->dimensions.end(), 1, std::multiplies<unsigned int>());
        this->tensor = std::shared_ptr<T[]>(new T[this->totalEntries], std::default_delete<T[]>());
        std::copy(src_tensor, src_tensor + this->totalEntries, this->tensor.get());
    }

    // Honestly I just know how to implement tensor plus, minus
    // I have not really found a way to implement tensor cross product.
    Tensor<T> operator+(const Tensor<T>& other) const {
        if (this->dimensions != other.getDim()){
            throw std::runtime_error("Cannot perform addition on two different tensors");
        }

        // Create new tensor instance
        Tensor<T> newTensor(this->dimensions);
        
        // Perform addition on this new tensor
        for (unsigned int i = 0; i < other.totalEntries; i++){
            newTensor.tensor[i] = this->tensor[i] + other.tensor[i];
        }

        return newTensor;
    }

    Tensor<T> operator-(const Tensor<T>& other) const {
        if (this->getDim() != other.getDim()){
            throw std::runtime_error("Cannot perform addition on two different tensors");
        }

        // Create new tensor instance
        Tensor<T> newTensor(this->getDim());

        // Perform subtraction on this new tensor
        for (unsigned int i = 0; i < other.totalEntries; i++){
            newTensor.tensor[i] = this->tensor[i] - other.tensor[i];
        }

        return newTensor;
    }

    Tensor<T> operator-() const {
        // Create new tensor instance
        Tensor<T> newTensor(this->getDim());

        // Perform invertion on this new tensor
        for (unsigned int i = 0; i < newTensor.getTotalEntries(); i++){
            newTensor.tensor[i] = -this->tensor[i];
        }

        return newTensor;
    }

    /**
     * Allowing people to use syntax like instanceOfTensor({0,1}) to access each entry
     * 
     * The order of value in entry is from the least to most significant dimension.
     * 
     * ALERT: if the entry is modified via this method like 
     *                instanceOfTensor1 = instanceOfTensor2
     *                instanceOfTensor1({8,9}) = 8;
     * The instanceOfTensor2 is also modified because they use the same shared pointer.
     */
    T& operator()(std::vector<unsigned int> posVec){
        return this->tensor[this->posVecToIndex(posVec)];
    }
    const T& operator()(std::vector<unsigned int> posVec) const{
        return this->tensor[this->posVecToIndex(posVec)];
    }
    
    // Get the total number of entries of this tensor, depending on the dimension size
    size_t getTotalEntries() const {
        return this->totalEntries;
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

    /**
     * Set the entire tensor's entries from src_tensor
     * The number of entries of src_tensor must match with this tensor.
     */
    void setTensor(const T *src_tensor, size_t num_entries){
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
     * Reset the dimension for the Tensor.
     * 
     * This does not change the relative position of each entry in the RAM but will change
     * the way the tensor understand each entry. The total number of entries of the new dimension
     * must be the same with the current total of entry.
     * 
     * No entry will be lost after this operation.
     */
    void reShape(const std::vector<unsigned int>& newDim){
        if ((unsigned int)std::accumulate(newDim.begin(), newDim.end(), 1, std::multiplies<unsigned int>()) != this->totalEntries){
            throw std::runtime_error("New dimension is invalid");
        }
        this->dimensions = newDim;
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

    // Inspired by toString() in Java
    std::string toString(){
        return this->toStringHelper(this->dimensions.end() - 1, 0);
    }
};


#endif
