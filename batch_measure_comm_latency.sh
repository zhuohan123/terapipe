#!/bin/bash
MODELS="gpt2-1hm,gpt2-3hm,gpt2-7hm,gpt2-1b,megatron-1b,megatron-2b,megatron-4b,megatron-8b,gpt3-1hm,gpt3-3hm,gpt3-7hm,gpt3-1b,gpt3-2b,gpt3-6b,gpt3-13b,gpt3-175b"

for N_GPU in `seq 2 8`; do
    echo "pipeline[n_node=1, model=$MODELS, n_gpus=$N_GPU]"
    ./measure_comm_latency.sh 1 $N_GPU pipeline "$MODELS" 10
    echo "allreduce[n_node=1, model=$MODELS, n_gpus=$N_GPU]"
    ./measure_comm_latency.sh 1 $N_GPU allreduce "$MODELS" 10
done

for N_GPU in `seq 1 8`; do
    echo "pipeline[n_node=2, model=$MODELS, n_gpus=$N_GPU]"
    ./measure_comm_latency.sh 2 $N_GPU pipeline "$MODELS" 10
    echo "allreduce[n_node=2, model=$MODELS, n_gpus=$N_GPU]"
    ./measure_comm_latency.sh 2 $N_GPU allreduce "$MODELS" 10
done
