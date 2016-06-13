LIBS=-lpthread -lgflags
FLAGS=-Ofast -std=c++11
CC=g++

all:
	$(CC) $(FLAGS) src/main.cpp $(LIBS) -o cyclades
