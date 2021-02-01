#!/bin/bash
./mpirun_dp_slices_evaluation.sh 24 8 gpt3-1b
./mpirun_dp_slices_evaluation.sh 40 8 gpt3-13b
./mpirun_dp_slices_evaluation.sh 48 8 gpt3-44b
./mpirun_dp_slices_evaluation.sh 48 8 gpt3-175b
