

#include <vector>

namespace IdioticML {    
namespace Algorithm {


unsigned int posVecToIndex(const std::vector<unsigned int>& pos, 
                           const std::vector<unsigned int>& dimensions, 
                           unsigned int totalEntries);

std::vector<unsigned int> sumOfCartesianProd(std::vector<std::vector<unsigned int>> sets);


unsigned int bigToLittle_endian(const unsigned char* bytes);
    
}
}




