#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <array>
#include <opencv2/opencv.hpp>



unsigned int bigToLittle_endian(const unsigned char* bytes){
    return (bytes[0] << 24) |
           (bytes[1] << 16) |
           (bytes[2] << 8)  |
           (bytes[3]);
}

class DataSet {
/**
 * Specifically used for loading and getting images of MNIST handwritten dataset.
 * 
 * Format of the MNIST image file:
 * 
 * Offset   Bytes   Description
 * 0        4       Magic number //What the fuck??
 * 4        4       Number of images
 * 8        4       Number of pixels of the height
 * 12       4       Number of pixels of the width
 * 16       1       pixel value
 * 17       1       pixel value
 * ..       ..      pixel value
 */
private:
    std::ifstream srcFile;
    unsigned int total;    // Total number of images
    unsigned int npHeight; // Number of pixels in height
    unsigned int npWidth;  // Number of pixels in width
    unsigned int npSize;   // Total number of pixels per image

    unsigned int nth_image;// Current nth image

public:
    
    DataSet(std::string path){
        this->srcFile.open(path, std::ios::binary);

        char buffer[4];

        // Set the total number of images
        this->srcFile.seekg(4);
        this->srcFile.read(buffer, 4);
        this->total = bigToLittle_endian((unsigned char *)buffer);
        
        // Set the height
        this->srcFile.seekg(8);
        this->srcFile.read(buffer, 4);
        this->npHeight = bigToLittle_endian((unsigned char *)buffer);

        // Set the width
        this->srcFile.seekg(12);
        this->srcFile.read(buffer, 4);
        this->npWidth = bigToLittle_endian((unsigned char *)buffer);

        // Set the total number of pixels
        this->npSize = this->npHeight * this->npWidth;

        // Reset the position to the beginning of the images' data
        this->srcFile.seekg(0);
        this->nth_image = 0;
    }

    ~DataSet(){
        this->srcFile.close();
    }

    /**
     * Get the current dataset index.
     * 
     * The returned number tell the index of the image that will be return if you call
     * nextImage().
     */
    unsigned int getCurrentImgIndex(){
        return this->nth_image;
    }

    void setCurrentImgIndex(unsigned int newIndex){
        this->nth_image = newIndex;
    }


    // Get the image
    // Useful for training process
    std::vector<uint8_t> nextImage(){
        std::vector<uint8_t> image;
        if (this->srcFile.eof()){
            return image;
        }

        // Set the file pointer to correct position
        this->srcFile.seekg(16 + this->nth_image * this->npSize);

        // Read the entire image into buffer.
        // Doing this can reduce File I/O
        unsigned char buffer[this->npSize];
        this->srcFile.read(buffer, this->npSize);

        // Upate the current image's index
        this->nth_image++;

        // Form image's vector
        image.assign(buffer, buffer + this->npSize);
        return image;
    }

    

    void displayImage(unsigned int nth){
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
};

int main(){

    DataSet dataset("MNIST/train/images");

    
   // std::vector<std::vector<uint8_t>> image = dataset.nextImage();

    dataset.displayImage(1);

    
    return 0;
}