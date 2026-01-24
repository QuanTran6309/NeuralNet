#include "adapter/arithmetic/ArthOpsHandler.hpp"
#include "adapter/arithmetic/CpuBackend.hpp"
#include "adapter/arithmetic/GpuBackend.cuh"

namespace IdioticML {

ArthOpsHandler::ArthOpsHandler(){
    this->cpuBackend = std::make_unique<CpuBackend>(Device(DeviceType::CPU));
    this->gpuBackend = std::make_unique<GpuBackend>(Device(DeviceType::GPU));   
}

ArthOpsHandler::~ArthOpsHandler(){
    
}

ArthOpsHandler& ArthOpsHandler::getInstance() {
    static ArthOpsHandler instance;
    return instance;
}








}