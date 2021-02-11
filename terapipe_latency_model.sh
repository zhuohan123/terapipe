#!/bin/bash

if [ "$#" -lt 4 ]; then echo "$(tput setaf 1)[ERROR]$(tput sgr 0) number of gpus, model name, number of steps, [extra args] required"; exit -1; fi
if [ "$#" -gt 5 ]; then echo "$(tput setaf 1)[ERROR]$(tput sgr 0) too many arguments: $#"; exit -1; fi

N_NODES=1
MODEL=$1
N_GPUS=$2 # per node
BATCH_SIZE=$3
MODEL_PARALLEL_SIZE=$N_GPUS
N_STEPS=$4
EXTRA_ARGS=$5

PYTHON_EXEC=$(which python)
PYTHON_SCRIPT=$(realpath -s terapipe_latency_model.py)
ROOT_DIR=$(dirname $(realpath -s ${0}))
source ${ROOT_DIR}/scripts/load_cluster_env.sh

# ${ROOT_DIR}/scripts/fornode fuser -k 7777/tcp
${ROOT_DIR}/scripts/fornode pkill python

echo ALL_IPADDR ${ALL_IPADDR[@]}
all_hosts=$(echo ${ALL_IPADDR[@]:0:$N_NODES} | sed 's/ /,/g')


# '--oversubscribe' enables MPI to run muliple processes per node.
for s in 0 1 2; do
  for p in $(ps aux | grep python | grep mixed | grep -v grep | awk '{print $2}'); do kill -9 $p; done
  sleep 5
  mpirun --mca btl_tcp_if_exclude lo,docker0 --mca oob_tcp_if_exclude lo,docker0 \
    --map-by ppr:$N_GPUS:node --oversubscribe -H $all_hosts \
        ${PYTHON_EXEC} ${PYTHON_SCRIPT} \
        $MY_IPADDR --port 7777 \
        --model-parallel-size ${MODEL_PARALLEL_SIZE} \
        --model ${MODEL} \
        --batch-size ${BATCH_SIZE} \
        --n-steps ${N_STEPS} \
        --use-mpi --sort-function $s \
        ${EXTRA_ARGS}
done
