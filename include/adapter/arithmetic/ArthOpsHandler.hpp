#ifndef ARTHOPSHANDLER
#define ARTHOPSHANDLER

#include "OpsBackend.hpp"
#include <memory>
namespace IdioticML{

class ArthOpsHandler
{
private:
    std::unique_ptr<OpsBackend> cpuBackend;
    std::unique_ptr<OpsBackend> gpuBackend;

    ArthOpsHandler();
    static ArthOpsHandler& getInstance();
public:
    ArthOpsHandler(const ArthOpsHandler&) = delete;    
    ArthOpsHandler& operator=(const ArthOpsHandler&) = delete;

    ~ArthOpsHandler();

    
};



}
#endif