#!/bin/bash
for i in 0 1 2 3 5; do
./mpirun_dp_slices_evaluation.sh 24 8 gpt3-1b $i
done

for i in `seq 0 7`; do
./mpirun_dp_slices_evaluation.sh 40 8 gpt3-13b $i
done

for i in 0 1 2; do
./mpirun_dp_slices_evaluation.sh 48 8 gpt3-44b $i
done

for i in 0 1; do
./mpirun_dp_slices_evaluation.sh 48 8 gpt3-175b $i
done

./mpirun_dp_slices_evaluation.sh 40 8 gpt3-13b-4096 0
./mpirun_dp_slices_evaluation.sh 40 8 gpt3-13b-6144 0
./mpirun_dp_slices_evaluation.sh 40 8 gpt3-13b-8192 0
