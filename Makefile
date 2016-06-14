LIBS=-lpthread -lgflags -fopenmp
FLAGS=-Ofast -std=c++11
CC=clang-omp++

all:
	$(CC) $(FLAGS) src/main.cpp $(LIBS) -o cyclades
