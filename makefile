CXX=g++

CXXFLAGS= -Wall -I include/
DEBUGFLAG=-g

BUILD_DIR=build
INCLUDE_DIR=include
DATA_DIR=data

INCLUDE_SRCS=$(shell find $(INCLUDE_DIR) -name '*.cpp')
INCLUDE_OBJ=$(INCLUDE_SRCS:%.cpp=$(BUILD_DIR)/%.o)

DATA_SRCS=$(shell find $(DATA_DIR) -name '*.cpp')
DATA_OBJ=$(DATA_SRCS:%.cpp=$(BUILD_DIR)/%.o)


OPENCV=$(shell pkg-config --cflags --libs opencv4)

all: main

main: $(INCLUDE_OBJ) $(DATA_OBJ) main.cpp
	$(CXX) $(CXXFLAGS) $(DEBUGFLAG) $^ -o main $(OPENCV) 

$(BUILD_DIR)/$(DATA_DIR)/%.o: $(DATA_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OPENCV) -c $< -o $@

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@