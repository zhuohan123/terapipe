#!/bin/bash
export LDFLAGS=-L/usr/local/cuda/lib64
rm _nccl.cpp
python setup.py build_ext --inplace
