# Includes for gflags only for the millenium mayhem
INCLUDES=-I/home/eecs/agnusmaximus/gflags/build/include/ -L/home/eecs/agnusmaximus/gflags/build/lib

# Require libraries
LIBS=-lpthread -lgflags -fopenmp

# Flags
FLAGS=-Ofast -std=c++11

# Select compiler (mac uses clang-omp++). Default is g++.
CC=g++
CLANG_OMP++ := $(shell command clang-omp++ --version 2> /dev/null)
ifdef CLANG_OMP++
    CC=clang-omp++
endif

all:
	$(CC) $(INCLUES) $(FLAGS) src/main.cpp $(LIBS) -o cyclades
