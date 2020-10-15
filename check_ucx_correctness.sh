#!/bin/bash

if [ "$#" -lt 2 ]; then echo "$(tput setaf 1)[ERROR]$(tput sgr 0) number of nodes, number of gpus per node required"; exit -1; fi
if [ "$#" -gt 2 ]; then echo "$(tput setaf 1)[ERROR]$(tput sgr 0) too many arguments: $#"; exit -1; fi

N_NODES=$1
N_GPUS=$2 # per node
MODEL=test-l512
N_SLICES=8
N_STEPS=10

PYTHON_EXEC=/home/ubuntu/anaconda3/envs/ucx/bin/python
PYTHON_SCRIPT=$(realpath -s test_transformer_ucx.py)
ROOT_DIR=$(dirname $(realpath -s ${0}))
source ${ROOT_DIR}/load_cluster_env.sh

# ${ROOT_DIR}/fornode fuser -k 7777/tcp
${ROOT_DIR}/fornode pkill python

PREV_ADDR=""
CHECKPOINT_PATH=${ROOT_DIR}/checkpoints/ckpt.pt

echo ALL_IPADDR ${ALL_IPADDR[@]}

WORLD_SIZE=$((N_NODES * N_GPUS))

PREV_PORT=0
i=0
for node_id in $(seq 0 $((N_NODES - 1))); do
  for gpu_id in $(seq 0 $((N_GPUS - 1))); do
    NODE_ADDR=${ALL_IPADDR[node_id]}
    MY_PORT=$((13300 + gpu_id))
    echo ${i} "=>" ${NODE_ADDR}:$MY_PORT
    if [ ${i} == 0 ]; then
      ssh -o StrictHostKeyChecking=no ${NODE_ADDR} \
         ${PYTHON_EXEC} ${PYTHON_SCRIPT} \
          --my-address ${NODE_ADDR} \
          --my-port ${MY_PORT} \
          --rank ${i} \
          --local-rank ${gpu_id} \
          --world-size ${WORLD_SIZE} \
          --check-correctness \
          --checkpoint-path ${CHECKPOINT_PATH} \
          --model ${MODEL} \
          --n-slices ${N_SLICES} \
          --n-steps ${N_STEPS} \
      &
    elif [ ${i} == $((WORLD_SIZE - 1)) ]; then
      ssh -o StrictHostKeyChecking=no ${NODE_ADDR} \
        ${PYTHON_EXEC} ${PYTHON_SCRIPT} \
          --prev-address ${PREV_ADDR} \
          --prev-port ${PREV_PORT} \
          --rank ${i} \
          --local-rank ${gpu_id} \
          --world-size ${WORLD_SIZE} \
          --check-correctness \
          --checkpoint-path ${CHECKPOINT_PATH} \
          --model ${MODEL} \
          --n-slices ${N_SLICES} \
          --n-steps ${N_STEPS} \
      &
    else
      ssh -o StrictHostKeyChecking=no ${NODE_ADDR} \
        ${PYTHON_EXEC} ${PYTHON_SCRIPT} \
          --my-address ${NODE_ADDR} \
          --my-port ${MY_PORT} \
          --prev-address ${PREV_ADDR} \
          --prev-port ${PREV_PORT} \
          --rank ${i} \
          --local-rank ${gpu_id} \
          --world-size ${WORLD_SIZE} \
          --check-correctness \
          --checkpoint-path ${CHECKPOINT_PATH} \
          --model ${MODEL} \
          --n-slices ${N_SLICES} \
          --n-steps ${N_STEPS} \
      &
    fi
    PREV_ADDR=${NODE_ADDR}
    PREV_PORT=${MY_PORT}
    i=$((i + 1))
  done
done
wait
