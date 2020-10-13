#!/bin/bash

mkdir -p log/

N_NODES=8
N_STEPS=10

for MODEL in "test-s1024" "test-s2048" "gpt2-1hm" "gpt2-3hm" "gpt2-7hm" "gpt2-1b" "megatron-1b" "megatron-2b" "megatron-4b" "megatron-8b" "gpt3-1hm" "gpt3-3hm" "gpt3-7hm" "gpt3-1b" "gpt3-2b" "gpt3-6b"; do
  for N_SLICES in 1 2 4 8 16; do
    ./run_ucx.sh $N_NODES $MODEL $N_SLICES $N_STEPS 2>&1 | tee log/seqpipe-2xlarge-$N_NODES-$MODEL-$N_SLICES-$N_STEPS.log
  done
done
