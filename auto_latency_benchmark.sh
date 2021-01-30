#!/bin/bash

# ./pipemegatron_latency_model.sh MODEL MODEL_PARALLEL_SIZE BATCH_SIZE N_STEPS EXTRA_ARGS

./pipemegatron_latency_model.sh gpt3-44b 8 16 10 --mixed-precision
./pipemegatron_latency_model.sh gpt3-44b 1 2 10 --mixed-precision

./pipemegatron_latency_model.sh gpt3-13b 8 32 10 --mixed-precision
./pipemegatron_latency_model.sh gpt3-13b 1 7 10 --mixed-precision

./pipemegatron_latency_model.sh gpt3-1b 8 72 10 --mixed-precision
./pipemegatron_latency_model.sh gpt3-1b 1 16 10 --mixed-precision

./pipemegatron_latency_model.sh gpt3-175b 8 2 10 --mixed-precision
./pipemegatron_latency_model.sh gpt3-175b 4 2 10 --mixed-precision
