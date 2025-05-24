CXX=g++

CXXFLAGS= -Wall -I include/
DEBUGFLAG=-g

BUILD_DIR=build
INCLUDE_DIR=include

INCLUDE_SRCS=$(shell find $(INCLUDE_DIR) -name '*.cc')
INCLUDE_OBJ=$(INCLUDE_SRCS:%.cc=$(BUILD_DIR)/%.o)


OPENCV=`pkg-config --cflags --libs opencv4`

all: main

main: $(INCLUDE_OBJ) main.cc
	$(CXX) $(CXXFLAGS) $(DEBUGFLAG) $(OPENCV) $^ -o main

$(BUILD_DIR)/%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@