#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <random>  

class Layer {
private:
    std::vector<float> entries;
public:
    // Construct by specific array
    Layer(std::vector<float> arr){
        this->entries = arr;
    }

    // Construct by size, randomly assign values
    Layer(unsigned int size){
        std::random_device rd;  
        std::mt19937 gen(rd()); 
        std::uniform_real_distribution<float> dist(0, 50);

        this->entries.resize(size);
        for (int i = 0; i < size; i++){
            this->entries[i] = dist(gen);
        }
    }

    ~Layer(){}

    void display(){
        std::ostringstream buffer;
        for (int i = 0; i < this->entries.size(); i++){
            buffer << std::fixed << std::setprecision(2) << this->entries[i] << ", ";
        }
        std::string output = buffer.str();
        if (!output.empty()){
            output.erase(output.size() - 2);
        }
        std::cout << '<' << output << '>' << std::endl;
    }

    unsigned int getSize(){
        return this->entries.size();
    }

};

class Model {
private:
    std::vector<float> weights;
    std::vector<float> biases;
public:
    Model(std::vector<Layer> layers){
        
        unsigned int total_weights = 1;
        unsigned int total_biased = 0;
        for (unsigned int i = 1; i < layers.size(); i++){
            
        }
    }
}


int main(void){
    

    
}