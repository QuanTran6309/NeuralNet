# A high performance C++ machine learning library
This is a light weight and high performance machine leaning library written entirely in C++. My aim is for people to use this library to run not just in their laptop or desktop but also in Iot devices which just have very limited memory.

## Table of content:
- [Roadmap](#roadmap)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)


## Usage
### 1. Tensor *(the most basic unit)*
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
        /**
         * It will returns 

         [0.000000, 1.000000, 2.000000],
         [3.000000, 4.000000, 5.000000]
        */
    }
    ```
    - from array pointer
    ```c++
    #include "tensor.hpp"
    
    int main(){
        // Dimension is {3, 1, 3} which indicate that this tensor is a 3D object with 3 columns, 1 row, 3 channels.
        float arr[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        Tensor<float> tensor(arr, {3, 1, 3});

        // Print out to visualize the tensor
        std::cout << tensor.toString() << std::endl;
        /**
         * It will returns 

         [[0.000000, 1.000000, 2.000000]],
         [[3.000000, 4.000000, 5.000000]],
         [[6.000000, 7.000000, 8.000000]]
        */
    }
    ```
    - or just dimension
    ```c++
    #include "tensor.hpp"
    
    int main(){
        Tensor<float> tensor({3, 1, 3});
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
        float arr[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        Tensor<float> tensor(arr, {3, 1, 3});
        
        // Let say you want to access the entry at second column, first row, and third channel.
        std::cout << tensor({1, 0, 2}) << std::endl;
        // returns 7

        return 0;
    }
    ```
- Tensor class also overload the (+ -). So if you want to perform tensor addition or minus, no need to call any method. I do not really know how to implement a generic formula for tensor cross product, so I can only implement cross product for class Matrix which derives from class Tensor.
    ```c++
    #include "tensor.hpp" 
    #include <iostream>
    int main() {
        float arr1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        Tensor<float> tensor1(arr1, {3, 1, 3});
        
        float arr2[] = {0, 4, 5, 6, 7, 8, 9, 10, 11};
        Tensor<float> tensor2(arr2, {3, 1, 3});

        Tensor<float> tensor = tensor1 + tensor2;
        Tensor<float> tensor = tensor1 - tensor2;
        Tensor<float> tensor = -tensor2;
    }
    ```
### 2. Matrix *(derive from class Tensor)*
- Also has + - overloaded.
- Also support matrix cross product with (*).


