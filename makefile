CXX=g++

CXXFLAGS= -I include/
DEBUGFLAG=-g

BUILD_DIR=build
SRC_DIR=src
DATA_DIR=data

CPP_SRC=$(shell find $(SRC_DIR) -name '*.cpp')
CPP_OBJ=$(CPP_SRC:%.cpp=$(BUILD_DIR)/%.o)

CU_SRC=$(shell find $(SRC_DIR) -name '*.cu')
CU_OBJ=$(CU_SRC:%.cu=$(BUILD_DIR)/%.o)

OPENCV=$(shell pkg-config --cflags --libs opencv4)

all: main

clean:
	rm -rf $(BUILD_DIR) main


main: $(CPP_OBJ) $(CU_OBJ)
	nvcc -Xcompiler -fPIC -shared -lcublas $(CXXFLAGS) $(DEBUGFLAG) $^ -o libtensor.so

$(BUILD_DIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	nvcc -Xcompiler -fPIC -lcublas $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	nvcc -Xcompiler -fPIC $(CXXFLAGS) -c $< -o $@


cuda: 
	nvcc -Xcompiler -fPIC -shared -I include/  include/algebra/tensor.cu include/cuda/cuda.cu  -o libtensor.so
	g++ main.cpp -L. -ltensor -I include/ -o main