#include "memory/device_adapter.hpp"

namespace IdioticML {

DeviceAdapter::DeviceAdapter(const Device& device) : device(device){}

DeviceType DeviceAdapter::getDeviceType(){
    return this->device.type;
}

}