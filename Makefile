LIBS=-lpthread -fopenmp -lgflags
FLAGS=-Ofast -std=c++11
CC=g++-5

all:
	$(CC) $(FLAGS) src/main.cpp $(LIBS) -o cyclades
