#!/bin/bash
if [ "$#" -lt 9 ]; then echo "$(tput setaf 1)[ERROR]$(tput sgr 0) number of nodes, number of gpus per node, model parallel size, pipeline parallel size, model name, batch size, number of batch slices, number of input slices, number of steps, [extra args] required"; exit -1; fi
if [ "$#" -gt 11 ]; then echo "$(tput setaf 1)[ERROR]$(tput sgr 0) too many arguments: $#"; exit -1; fi
N_NODES=$1
N_GPUS=$2 # per node
MODEL_PARALLEL_SIZE=$3
PIPELINE_PARALLEL_SIZE=$4
MODEL=$5
BATCH_SIZE=$6
N_BATCH_SLICES=$7
N_INPUT_SLICES=$8
N_STEPS=$9
EXTRA_ARGS=${@:10}

PYTHON_EXEC=/home/ubuntu/anaconda3/bin/python
PYTHON_SCRIPT=$(realpath -s test_transformer_terapipe.py)
ROOT_DIR=$(dirname $(realpath -s ${0}))
source ${ROOT_DIR}/scripts/load_cluster_env.sh

# ${ROOT_DIR}/scripts/fornode fuser -k 7777/tcp
${ROOT_DIR}/scripts/fornode "pgrep -fl python | awk '!/batch_test\.py/{print $1}' | xargs sudo kill"

echo ALL_IPADDR ${ALL_IPADDR[@]}
all_hosts=$(echo ${ALL_IPADDR[@]:0:$N_NODES} | sed 's/ /,/g')

# '--oversubscribe' enables MPI to run muliple processes per node.
/home/ubuntu/anaconda3/bin/mpirun --mca btl_tcp_if_exclude lo,docker0 --mca oob_tcp_if_exclude lo,docker0 \
    --map-by ppr:$N_GPUS:node --oversubscribe -H $all_hosts \
        ${PYTHON_EXEC} ${PYTHON_SCRIPT} \
        $MY_IPADDR -p 7777 \
        --model-parallel-size ${MODEL_PARALLEL_SIZE} \
        --pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE} \
        --model ${MODEL} \
        --batch-size ${BATCH_SIZE} \
        --n-batch-slices ${N_BATCH_SLICES} \
        --n-input-slices ${N_INPUT_SLICES} \
        --n-steps ${N_STEPS} \
        --use-mpi \
        ${EXTRA_ARGS}
