#!/bin/bash

if [ "$#" -lt 5 ]; then echo "$(tput setaf 1)[ERROR]$(tput sgr 0) number of nodes, number of gpus per node, model name, number of slices, number of steps required"; exit -1; fi
if [ "$#" -gt 5 ]; then echo "$(tput setaf 1)[ERROR]$(tput sgr 0) too many arguments: $#"; exit -1; fi

N_NODES=$1
N_GPUS=$2 # per node
MODEL=$3
N_SLICES=$4
N_STEPS=$5

PYTHON_EXEC=python
PYTHON_SCRIPT=$(realpath -s test_transformer_nccl.py)
ROOT_DIR=$(dirname $(realpath -s ${0}))
source ${ROOT_DIR}/scripts/load_cluster_env.sh

# ${ROOT_DIR}/scripts/fornode fuser -k 7777/tcp
${ROOT_DIR}/scripts/fornode pkill python

PREV_ADDR=""

echo ALL_IPADDR ${ALL_IPADDR[@]}

WORLD_SIZE=$((N_NODES * N_GPUS))

PREV_PORT=0
i=0
for node_id in $(seq 0 $((N_NODES - 1))); do
  for gpu_id in $(seq 0 $((N_GPUS - 1))); do
    NODE_ADDR=${ALL_IPADDR[node_id]}
    echo ${i} "=>" ${NODE_ADDR}
    ssh -o StrictHostKeyChecking=no ${NODE_ADDR} \
        ${PYTHON_EXEC} ${PYTHON_SCRIPT} \
        --rank ${i} \
        --local-rank ${gpu_id} \
        --world-size ${WORLD_SIZE} \
        --model ${MODEL} \
        --n-slices ${N_SLICES} \
        --n-steps ${N_STEPS} &
    i=$((i + 1))
  done
done
wait
