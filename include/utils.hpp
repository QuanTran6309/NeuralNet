#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <random>

namespace Utils {

    // Produce the cartesian product and sum each element within each set
    std::vector<unsigned int> sumOfCartesianProd(std::vector<std::vector<unsigned int>> sets);

    // Random utility
    float getRealRandom(float min, float max);
    int getIntRandom(int min, int max);
    template<typename T>
    T getRandom(T min, T max){
        static std::random_device rd;  
        static std::mt19937 gen(rd()); 
        std::uniform_real_distribution<T> dist(min, max);
        return dist(gen);
    }

    unsigned int bigToLittle_endian(const unsigned char* bytes);
}


#endif