#!/bin/bash

if [ "$#" -lt 4 ]; then echo "$(tput setaf 1)[ERROR]$(tput sgr 0) number of nodes, model name, number of slices, number of steps required"; exit -1; fi
if [ "$#" -gt 4 ]; then echo "$(tput setaf 1)[ERROR]$(tput sgr 0) too many arguments: $#"; exit -1; fi

N_NODES=$1
MODEL=$2
N_SLICES=$3
N_STEPS=$4

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
        --n-slices ${N_SLICES} \
        --n-steps ${N_STEPS} \
    &
  elif [ ${i} == $((N_NODES - 1)) ]; then
    ssh -o StrictHostKeyChecking=no ${NODE_ADDR} \
      ${PYTHON_EXEC} ${PYTHON_SCRIPT} \
        --prev-address ${PREV_ADDR} \
        --prev-port 7777 \
        --rank ${i} \
        --world-size ${N_NODES} \
        --model $MODEL \
        --n-slices ${N_SLICES} \
        --n-steps ${N_STEPS} \
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
        --n-slices ${N_SLICES} \
        --n-steps ${N_STEPS} \
    &
  fi
  PREV_ADDR=${NODE_ADDR}
done
wait
