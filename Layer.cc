#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <random>  
#include <type_traits>



template<typename T>
class Tensor {
    static_assert(std::is_arithmetic<T>::value, "Tensor only takes numeric values");

private:
    std::vector<T> tensor;
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
        unsigned int total_entries = 1;
        for (unsigned int dim : dimensions){
            if (dim == 0){
                throw std::runtime_error("Dimension of a tensor cannot be 0");
            }
            total_entries *= dim;
        }
        this->tensor.resize(total_entries);
        
        // Randomly assign the entries.
        std::random_device rd;  
        std::mt19937 gen(rd()); 
        std::uniform_real_distribution<float> dist(-100, 100);
        for (size_t i = 0; i < this->tensor.size(); i++){
            this->tensor[i] = dist(gen);
        }
        for (T entry : this->tensor){
            std::cout << entry << ", ";
        }
        std::cout << std::endl;
    }

    T getEntry(std::vector<unsigned int> entry){
        if (entry.size() > this->tensor.size()){
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
    
};


class Layer : public Tensor<float>{
private:
    
public:

    // Construct by size, randomly assign values
    Layer(unsigned int size) : Tensor<float>({1, size}){}

    
};



class Model {
private:
    std::vector<float> weights;
    std::vector<float> biases;
    std::vector<Layer> layers;
    float learn_rate = 0.01;
public:
    /**
     * Random constructor
     * 
     * This way to creating a model will randomly assign the value for 
     */
    Model(std::vector<Layer> layers){
        if (layers.size() <= 1){
            throw std::runtime_error("There must be more than 1 layer for a model\n");
        }

        // Set the size for weights and biases matrices
        unsigned int total_weights = 0;
        unsigned int total_biases = 0;
        for (unsigned int i = 1; i < layers.size(); i++){
            total_weights += (layers[i - 1].getSize() * layers[i].getSize());
            total_biases += layers[i].getSize();
        }
        weights.resize(total_weights);
        biases.resize(total_biases);

        // The number of weights is always higher than the number of biases
        // So can group their assignment into a single for loop
        
        for (int i = 0; i < weights.size(); i++){
            
        }

        // Set the layers
        this->layers = layers;
    }

    

    
};


int main(void){
    Tensor<float> weights({2, 3, 4});
    
    
}