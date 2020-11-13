#!/bin/bash

if [ "$#" -lt 7 ]; then echo "$(tput setaf 1)[ERROR]$(tput sgr 0) number of nodes, number of gpus per node, model parallel size, pipeline parallel size, model name, number of slices, number of steps, [extra args] required"; exit -1; fi
if [ "$#" -gt 8 ]; then echo "$(tput setaf 1)[ERROR]$(tput sgr 0) too many arguments: $#"; exit -1; fi

N_NODES=$1
N_GPUS=$2 # per node
MODEL_PARALLEL_SIZE=$3
PIPELINE_PARALLEL_SIZE=$4
MODEL=$5
N_SLICES=$6
N_STEPS=$7
EXTRA_ARGS=$8

PYTHON_EXEC=$(which python)
PYTHON_SCRIPT=$(realpath -s test_transformer_pipemegatron.py)
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
        $MY_IPADDR -p 7777 \
        --model-parallel-size ${MODEL_PARALLEL_SIZE} \
        --pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE} \
        --model ${MODEL} \
        --n-slices ${N_SLICES} \
        --n-steps ${N_STEPS} \
        --use-mpi \
        ${EXTRA_ARGS}
