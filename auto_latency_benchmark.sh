#!/bin/bash

# ./pipemegatron_latency_model.sh MODEL MODEL_PARALLEL_SIZE BATCH_SIZE N_STEPS EXTRA_ARGS
./pipemegatron_latency_model.sh gpt3-1b 8 128 10 --mixed-precision
./pipemegatron_latency_model.sh gpt3-1b 4 64 10 --mixed-precision
./pipemegatron_latency_model.sh gpt3-1b 2 32 10 --mixed-precision
./pipemegatron_latency_model.sh gpt3-1b 1 16 10 --mixed-precision

./pipemegatron_latency_model.sh gpt3-13b 8 48 10 --mixed-precision
./pipemegatron_latency_model.sh gpt3-13b 4 24 10 --mixed-precision
./pipemegatron_latency_model.sh gpt3-13b 2 12 10 --mixed-precision
./pipemegatron_latency_model.sh gpt3-13b 1 7 10 --mixed-precision

./pipemegatron_latency_model.sh gpt3-44b 8 16 10 --mixed-precision
./pipemegatron_latency_model.sh gpt3-44b 4 8 10 --mixed-precision
./pipemegatron_latency_model.sh gpt3-44b 2 4 10 --mixed-precision
./pipemegatron_latency_model.sh gpt3-44b 1 2 10 --mixed-precision

./pipemegatron_latency_model.sh gpt3-175b 8 2 10 --mixed-precision
./pipemegatron_latency_model.sh gpt3-175b 4 2 10 --mixed-precision
