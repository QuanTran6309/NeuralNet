#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <random>  
#include <type_traits>


struct Range {
    int start; // inlcusive
    int end;   // exclusive
};


template<typename T>
class Tensor {
    static_assert(std::is_arithmetic<T>::value, "Tensor only takes numeric values");

private:
    T *tensor;
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
        this->tensor = new T[total_entries];
    }
    
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

    // Must free the allocated memory for the tensor.
    ~Tensor(){
        delete[] this->tensor;
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

    // Get just a portion of a tensor,
    // Tensor slicing
    Tensor<T> getTensor(std::vector<Range> bounds){

        // Later add more condition here in case bounds is out of bound

        // Calculate the total number of entries, dimensions, and bounds for a new tensor
        unsigned int total_entries = 1;
        unsigned int prevSize = 1;
        std::vector<unsigned int> newDim;
        std::vector<unsigned int> indices; // Determine at which location on the current tensor to be copied to the new tensor
        for (unsigned int i = 0; i < bounds.size(); i++){
            unsigned int range = bounds[i].end - bounds[i].start;
            total_entries *= range;
            newDim.push_back(range);

            // Form the index for later getting and assigning new tensor value
            std::vector<unsigned int> index;
            for (unsigned int j = bounds[i].start; j < bounds[i].end; j++){
                index.push_back(j * prevSize);
            }
            if (indices.size() == 0){
                indices = index;
            }
            else {
                std::vector<unsigned int> buffer;
                for (unsigned int idx : index){
                    for (unsigned int idc : indices){
                        buffer.push_back(idx + idc);
                    }
                }
                indices = buffer;
            }
            prevSize *= this->dimensions[i];
        }

        // Create and get new tensor
        Tensor<T> newTensor(newDim);
        T *newTensor_ptr = newTensor.getTensor();

        std::cout << "Printing the indices" << std::endl;
        for (int i = 0; i < indices.size(); i++){
            std::cout << indices[i] << "=>" << this->tensor[indices[i]] << std::endl;
        }
        

        // Copy value to new tensor.
        unsigned int itr = 0;
        while (itr < total_entries){
            newTensor_ptr[itr] = this->tensor[indices[itr]];
            itr++;
        }
        return newTensor;
    }

    std::vector<unsigned int> cartesian_sum(std::vector<std::vector<unsigned int>> vecs, unsigned int depth = 0){
        

    }

    // Get the entire tensor in a form of a 1D array
    // This can be dangerous, but necessary for saving memory.
    T *getTensor(){
        return this->tensor;
    }

    // Get the size of the tensor
    std::vector<unsigned int> getDim(){
        return this->dimensions;
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
    Tensor<float> matrix({4, 3});
    matrix.randomlyInit();
    
    float *mat = matrix.getTensor();
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 4; j++){
            std::cout << mat[i * 4 + j] << " ";
        }
        std::cout << std::endl;
    }
    
    Tensor<float> subtensor = matrix.getTensor({
                                                {1,3}, 
                                                {0,3}
                                            });

    std::cout << "Printing new subtensor" << std::endl;
    
    float *tensor = subtensor.getTensor();
    std::vector<unsigned int> dim = subtensor.getDim();
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 2; j++){
            std::cout << tensor[i * 2 + j] << " ";
        }
        std::cout << std::endl;    
    }
    

    return 0;
}