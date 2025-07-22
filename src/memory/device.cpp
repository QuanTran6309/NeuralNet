#include "memory/device.hpp"

namespace IdioticML {

// Device struct
Device::Device(DeviceType type, int id) : type(type), id(id){}

Device Device::cpu(){ 
    return Device(DeviceType::CPU); 
}
Device Device::gpu(int id){ 
    return Device(DeviceType::GPU, id); 
}



}