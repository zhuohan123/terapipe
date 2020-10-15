#!/bin/bash

if [ "$#" -lt 1 ]; then echo "$(tput setaf 1)[ERROR]$(tput sgr 0) number of nodes, model name, number of steps required"; exit -1; fi
if [ "$#" -gt 1 ]; then echo "$(tput setaf 1)[ERROR]$(tput sgr 0) too many arguments: $#"; exit -1; fi

N_NODES=$1
MODEL=$2
N_STEPS=$3

ROOT_DIR=$(dirname $(realpath -s ${0}))
all_hosts=$(echo ${ALL_IPADDR[@]:0:$N_NODES} | sed 's/ /,/g')
source ${ROOT_DIR}/load_cluster_env.sh

# ${ROOT_DIR}/fornode fuser -k 7777/tcp
${ROOT_DIR}/fornode pkill python

mpirun --mca btl_tcp_if_include ens3 --map-by ppr:1:node -H $all_hosts \
  python test_megatron_distributed \
    $MY_IPADDR \
    -p 7777 \
    --model $MODEL \
    --n-steps $N_STEPS

