# A high performance C++ machine learning library
This is a light weight and high performance machine leaning library written entirely in C++. My aim is for people to use this library to run not just in their laptop or desktop but also in Iot devices which just have very limited memory.

## Table of content:
- [Roadmap](#roadmap)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)


## Usage
1. Tensor
- Mimic the multi-dimensional tensor just like in algebra.
- You can have as many dimensions as you want.
- It is a template class, so you have to specify the data type for the entry.
- Tensor class has method toString() which is inspired by Java toString(). toString() returns a std::string which visualize the Tensor. The style of the returned string is inspired by Pytorch tensor.
- Dimension of Tensor is a vector. Index of that vector tells the order of dimension.
    ```
    {1, 3, 4}    // 1 column, 3 rows, 4 channel
    {1, 3, 4, 5} // 1 column, 3 rows, 4 channel, 5 <whatever you call the 4th dimension is>
    ```
- Create an instance of tensor
    - from std::vector
    ```c++
    #include "tensor.hpp"
    
    int main(){
        // Dimension is {3, 2} which indicate that this tensor is basically a matrix with 3 columns and 2 rows.
        Tensor<float> tensor({0, 1, 2, 3, 4, 5}, {3, 2});

        // Print out to visualize the tensor
        std::cout << tensor.toString() << std::endl;
    }
    ```
    

- Tensor uses shared pointer to keep track of the entries, so any tensor assignment using '=' does not create a separate chunk of entries but shared accross both instances.
    ```c++
    #include "tensor.hpp" 
    int main() {
        Tensor<int> tensor = anotherInstanceOfTensor;
        return 0;
    }
    ```
 - Access each entry of a tensor using a position vector.
     ```c++
    #include "tensor.hpp" 
    #include <iostream>
    int main() {
        Tensor<int> tensor()
        

        return 0;
    }
    ```

