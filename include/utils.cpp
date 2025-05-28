#include "utils.hpp"

#include <random>

namespace Utils {

    // Produce the cartesian product and sum each element within each set
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


    float getRealRandom(float min, float max){
        static std::random_device rd;  
        static std::mt19937 gen(rd()); 
        std::uniform_real_distribution<float> dist(min, max);
        return dist(gen);
    }

    int getIntRandom(int min, int max){
        static std::random_device rd;  
        static std::mt19937 gen(rd()); 
        std::uniform_int_distribution<int> dist(min, max);
        return dist(gen);
    }


    unsigned int bigToLittle_endian(const unsigned char* bytes){
        return (bytes[0] << 24) |
               (bytes[1] << 16) |
               (bytes[2] << 8)  |
               (bytes[3]);
    }
    
}


