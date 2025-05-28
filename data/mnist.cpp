#include "mnist.hpp"
#include <opencv2/opencv.hpp>
#include "utils.hpp"

namespace DataEngine {

MNIST::MNIST(std::string path){
    this->srcFile.open(path, std::ios::binary);
    if (!this->srcFile.is_open()){
        throw std::runtime_error("Cannot open MNIST file at " + path);
    }

    char buffer[4];

    // Set the total number of images
    this->srcFile.seekg(4);
    this->srcFile.read(buffer, 4);
    this->total = Utils::bigToLittle_endian((unsigned char *)buffer);
    
    // Set the height
    this->srcFile.seekg(8);
    this->srcFile.read(buffer, 4);
    this->npHeight = Utils::bigToLittle_endian((unsigned char *)buffer);

    // Set the width
    this->srcFile.seekg(12);
    this->srcFile.read(buffer, 4);
    this->npWidth = Utils::bigToLittle_endian((unsigned char *)buffer);

    // Set the total number of pixels
    this->npSize = this->npHeight * this->npWidth;

    // Reset the position to the beginning of the images' data
    this->srcFile.seekg(0);
}

MNIST::~MNIST(){
    this->srcFile.close();
}

unsigned int MNIST::getTotalImages(){
    return this->total;
}

unsigned int MNIST::getTotalPixelsPerImage(){
    return this->npSize;
}

// Get image by index
Tensor<unsigned char> MNIST::getImage(unsigned int index){
    if (index >= this->total){
        throw std::runtime_error("Index out of bound");
    }

    // Set the file pointer to correct position
    this->srcFile.seekg(16 + index * this->npSize);

    // Read the entire image into buffer.
    // Doing this can reduce File I/O
    unsigned char buffer[this->npSize];
    this->srcFile.read((char *)buffer, this->npSize);
    
    // Form tensor : matrix
    return Tensor<unsigned char>(buffer, {this->npWidth, this->npHeight});
}

void MNIST::displayImage(unsigned int nth){
    if (nth >= this->total){
        return;
    }

    this->srcFile.seekg(16 + nth * this->npSize);

    char buffer[this->npSize];
    this->srcFile.read(buffer, this->npSize);


    std::string image_name = "MNIST at index " + nth;
    cv::Mat image(this->npHeight, this->npWidth, CV_8UC1, buffer);
    cv::imshow(image_name, image);
    cv::waitKey(0);
}



}