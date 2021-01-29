#!/bin/bash
N_NODES=2
N_GPUS=8 # per node

PYTHON_EXEC=$(which python)
PYTHON_SCRIPT=$(realpath -s p2p_comm_latency.py)
ROOT_DIR=$(dirname $(realpath -s ${0}))
source ${ROOT_DIR}/scripts/load_cluster_env.sh

${ROOT_DIR}/scripts/fornode pkill python

echo ALL_IPADDR ${ALL_IPADDR[@]}
all_hosts=$(echo ${ALL_IPADDR[@]:0:$N_NODES} | sed 's/ /,/g')

# '--oversubscribe' enables MPI to run muliple processes per node.
mpirun --mca btl_tcp_if_exclude lo,docker0 --mca oob_tcp_if_exclude lo,docker0 \
    --map-by ppr:$N_GPUS:node --oversubscribe -H $all_hosts \
        ${PYTHON_EXEC} ${PYTHON_SCRIPT}
