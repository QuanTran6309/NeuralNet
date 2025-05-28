#include <fstream>
#include "tensor.hpp"


namespace DataEngine
{
class MNIST {
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

public:

    // Take a path to the MNIST file by Yann LeCun
    MNIST(std::string path);

    // Must delete the file pointer
    ~MNIST();

    // Get the total number of images in the dataset.
    unsigned int getTotalImages();

    // Get the total number of pixels per image
    unsigned int getTotalPixelsPerImage();

    // Get image by index
    Tensor<unsigned char> getImage(unsigned int index);

    
    void displayImage(unsigned int nth);
};

}
