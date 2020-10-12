#!/bin/bash

if [ "$#" -lt 2 ]; then echo "$(tput setaf 1)[ERROR]$(tput sgr 0) number of nodes, model name required"; exit -1; fi
if [ "$#" -gt 2 ]; then echo "$(tput setaf 1)[ERROR]$(tput sgr 0) too many arguments: $#"; exit -1; fi

N_NODES=$1
MODEL=$2

PYTHON_EXEC=/home/ubuntu/anaconda3/envs/ucx/bin/python
PYTHON_SCRIPT=$(realpath -s test_transformer_ucx.py)
ROOT_DIR=$(dirname $(realpath -s ${0}))
source ${ROOT_DIR}/load_cluster_env.sh

# ${ROOT_DIR}/fornode fuser -k 7777/tcp
${ROOT_DIR}/fornode pkill python

PREV_ADDR=""

echo ALL_IPADDR ${ALL_IPADDR[@]}

for i in $(seq 0 $((N_NODES - 1))); do
  NODE_ADDR=${ALL_IPADDR[i]}
  echo ${i} "=>" ${NODE_ADDR}
  if [ ${i} == 0 ]; then
    ssh -o StrictHostKeyChecking=no ${NODE_ADDR} \
       ${PYTHON_EXEC} ${PYTHON_SCRIPT} \
        --my-address ${NODE_ADDR} \
        --my-port 7777 \
        --rank ${i} \
        --world-size ${N_NODES} \
        --model $MODEL \
    &
  elif [ ${i} == $((N_NODES - 1)) ]; then
    ssh -o StrictHostKeyChecking=no ${NODE_ADDR} \
      ${PYTHON_EXEC} ${PYTHON_SCRIPT} \
        --prev-address ${PREV_ADDR} \
        --prev-port 7777 \
        --rank ${i} \
        --world-size ${N_NODES} \
        --model $MODEL \
    &
  else
    ssh -o StrictHostKeyChecking=no ${NODE_ADDR} \
      ${PYTHON_EXEC} ${PYTHON_SCRIPT} \
        --my-address ${NODE_ADDR} \
        --my-port 7777 \
        --prev-address ${PREV_ADDR} \
        --prev-port 7777 \
        --rank ${i} \
        --world-size ${N_NODES} \
        --model $MODEL \
    &
  fi
  PREV_ADDR=${NODE_ADDR}
done
wait