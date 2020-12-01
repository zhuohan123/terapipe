#!/bin/bash

if [ "$#" -lt 3 ]; then echo "$(tput setaf 1)[ERROR]$(tput sgr 0) number of gpus, model name, number of steps, [extra args] required"; exit -1; fi
if [ "$#" -gt 4 ]; then echo "$(tput setaf 1)[ERROR]$(tput sgr 0) too many arguments: $#"; exit -1; fi

N_NODES=1
N_GPUS=$1 # per node
MODEL_PARALLEL_SIZE=$N_GPUS
MODEL=$2
N_STEPS=$3
EXTRA_ARGS=$4

PYTHON_EXEC=$(which python)
PYTHON_SCRIPT=$(realpath -s pipemegatron_latency_model.py)
ROOT_DIR=$(dirname $(realpath -s ${0}))
source ${ROOT_DIR}/scripts/load_cluster_env.sh

# ${ROOT_DIR}/scripts/fornode fuser -k 7777/tcp
${ROOT_DIR}/scripts/fornode pkill python

echo ALL_IPADDR ${ALL_IPADDR[@]}
all_hosts=$(echo ${ALL_IPADDR[@]:0:$N_NODES} | sed 's/ /,/g')

# '--oversubscribe' enables MPI to run muliple processes per node.
mpirun --mca btl_tcp_if_exclude lo,docker0 --mca oob_tcp_if_exclude lo,docker0 \
    --map-by ppr:$N_GPUS:node --oversubscribe -H $all_hosts \
        ${PYTHON_EXEC} ${PYTHON_SCRIPT} \
        $MY_IPADDR --port 7777 \
        --model-parallel-size ${MODEL_PARALLEL_SIZE} \
        --model ${MODEL} \
        --n-steps ${N_STEPS} \
        --use-mpi \
        ${EXTRA_ARGS}
