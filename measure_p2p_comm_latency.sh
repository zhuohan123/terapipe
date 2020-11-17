#!/bin/bash

if [ "$#" -lt 1 ]; then echo "$(tput setaf 1)[ERROR]$(tput sgr 0) requires payload size"; exit -1; fi
if [ "$#" -gt 1 ]; then echo "$(tput setaf 1)[ERROR]$(tput sgr 0) too many arguments: $#"; exit -1; fi

N_NODES=2
N_GPUS=2 # per node
PAYLOAD_SIZE=$1

PYTHON_EXEC=$(which python)
PYTHON_SCRIPT=$(realpath -s p2p_comm_latency.py)
ROOT_DIR=$(dirname $(realpath -s ${0}))
source ${ROOT_DIR}/scripts/load_cluster_env.sh

# ${ROOT_DIR}/scripts/fornode fuser -k 7777/tcp
${ROOT_DIR}/scripts/fornode pkill python

echo ALL_IPADDR ${ALL_IPADDR[@]}
all_hosts=$(echo ${ALL_IPADDR[@]:0:$N_NODES} | sed 's/ /,/g')

# '--oversubscribe' enables MPI to run muliple processes per node.
mpirun --mca btl_tcp_if_exclude lo,docker0 --mca oob_tcp_if_exclude lo,docker0 \
    --map-by ppr:$N_GPUS:node --oversubscribe -H $all_hosts \
        ${PYTHON_EXEC} ${PYTHON_SCRIPT} $PAYLOAD_SIZE
