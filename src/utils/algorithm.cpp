
#include "utils/algorithm.hpp"
#include <stdexcept>

namespace IdioticML {    
namespace Algorithm {


unsigned int posVecToIndex(const std::vector<unsigned int>& pos, 
                           const std::vector<unsigned int>& dimensions, 
                           unsigned int totalEntries) 
{
    if (pos.size() > dimensions.size() || pos.size() == 0){
        throw std::runtime_error("Position vector is invalid");
    }
    unsigned int prevSize = 1;
    unsigned int index = 0;
    for (unsigned int i = 0; i < pos.size(); i++){
        index += (pos[i] * prevSize);
        prevSize *= dimensions[i];
    }
    if (index >= totalEntries){
        throw std::runtime_error("Index out of bound");
    }
    return index;
}

std::vector<unsigned int> sumOfCartesianProd(std::vector<std::vector<unsigned int>> sets){
    if (sets.size() == 1){
        return sets[0];
    }
    std::vector<unsigned int> result;
    std::vector<std::vector<unsigned int>> subSet(sets.begin() + 1, sets.end());
    std::vector<unsigned int> deeperSet = sumOfCartesianProd(subSet);
    for (unsigned int curr : sets[0]){
        for (unsigned int deeper : deeperSet){
            result.push_back(curr + deeper);
        }
    }
    return result;
}

unsigned int bigToLittle_endian(const unsigned char* bytes){
    return (bytes[0] << 24) |
            (bytes[1] << 16) |
            (bytes[2] << 8)  |
            (bytes[3]);
}
    
}
}

