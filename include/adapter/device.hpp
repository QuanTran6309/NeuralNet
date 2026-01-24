
#ifndef DEVICE
#define DEVICE

#include <cstdint>
#include "algebra/tensortypes.hpp"


/**
 * NOTE: The word "Device" in this file does not only refer to the GPU but both GPU and CPU.
 */

namespace IdioticML {

struct Device {
    DeviceType type;
    int id; // For multiple GPU architecture

    Device (DeviceType type = DeviceType::CPU, int id = 0);

    static Device cpu();
    static Device gpu(int id = 0);
};

}



#endif