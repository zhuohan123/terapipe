#!/bin/bash
rm _nccl.cpp
python setup.py build_ext --inplace
