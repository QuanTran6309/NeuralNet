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

// Random wraper
template<typename T>
struct Randomizer{
    static_assert(std::is_arithmetic<T>::value, "Template parameter must be a primitive type");
    T (* randFunc)(T, T);
    T max;
    T min;
    Randomizer(T (* randFunc)(T, T), T max, T min) : randFunc(randFunc), max(max), min(min){}
    T generate(){
        return randFunc(min, max);
    }
};

unsigned int bigToLittle_endian(const unsigned char* bytes);
}


#endif