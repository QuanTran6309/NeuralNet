CXX=g++

CXXFLAGS= -Wall -I include/
DEBUGFLAG=-g

BUILD_DIR=build
INCLUDE_DIR=include

INCLUDE_SRCS=$(shell find $(INCLUDE_DIR) -name '*.cpp')
INCLUDE_OBJ=$(INCLUDE_SRCS:%.cpp=$(BUILD_DIR)/%.o)


OPENCV=`pkg-config --cflags --libs opencv4`

all: main

main: $(INCLUDE_OBJ) main.cpp
	$(CXX) $(CXXFLAGS) $(DEBUGFLAG) $(OPENCV) $^ -o main

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@