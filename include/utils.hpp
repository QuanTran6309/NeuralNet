#ifndef UTILS_H
#define UTILS_H

#include <vector>

namespace Utils {

    // Produce the cartesian product and sum each element within each set
    std::vector<unsigned int> sumOfCartesianProd(std::vector<std::vector<unsigned int>> sets);

    // Random utility
    float getRealRandom(float min, float max);
    int getIntRandom(int min, int max);

    unsigned int bigToLittle_endian(const unsigned char* bytes);
}


#endif