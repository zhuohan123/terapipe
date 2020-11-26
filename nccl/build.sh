#!/bin/bash
export LDFLAGS="-L/usr/local/cuda-11.0/efa/lib -L/usr/local/cuda-11.0/lib -L/usr/local/cuda-11.0/lib64 -L/usr/local/cuda-11.0 -L/opt/amazon/efa/lib -L/opt/amazon/openmpi/lib $LDFLAGS"
rm _nccl.cpp
python setup.py build_ext --inplace
