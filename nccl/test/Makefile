# We assumes CUDA version 10.2
LDFLAGS = -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcudart -L/usr/local/cuda/lib/ -lnccl

CXX = g++
#CPPFLAGS += `pkg-config --cflags mpi` # MPICXX keeps its own headers.
CXXFLAGS += -std=c++11 -O2 -g -fPIC
MPICXX?=mpicxx

all: round_trip_test

round_trip_test: round_trip_test.cc
	$(MPICXX) $^ $(LDFLAGS) -o $@
clean:
	rm -f round_trip_test.o round_trip_test
