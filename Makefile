# Require libraries
LIBS=-lpthread -lgflags -fopenmp

# Includes for gflags only for the millenium mayhem
if [ -d "/home/eecs/agnusmaximus/gflags/build/include" ]; then \
	LIBS += -I/home/eecs/agnusmaximus/gflags/build/include \
fi
if [ -d "/home/eecs/agnusmaximus/gflags/build/lib" ]; then \
	LIBS += -L/home/eecs/agnusmaximus/gflags/build/lib \
fi

# Flags
FLAGS=-Ofast -std=c++11

# Select compiler (mac uses clang-omp++). Default is g++.
CC=g++
CLANG_OMP++ := $(shell command clang-omp++ --version 2> /dev/null)
ifdef CLANG_OMP++
    CC=clang-omp++
endif

all:
	$(CC) $(FLAGS) src/main.cpp $(LIBS) -o cyclades
