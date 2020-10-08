#!/bin/bash

if [ "$#" -lt 1 ]; then echo "$(tput setaf 1)[ERROR]$(tput sgr 0) number of nodes required"; exit -1; fi
if [ "$#" -gt 1 ]; then echo "$(tput setaf 1)[ERROR]$(tput sgr 0) too many arguments: $#"; exit -1; fi

N_NODES=$1

PYTHON_EXEC=/home/ubuntu/anaconda3/envs/ucx/bin/python
PYTHON_SCRIPT=$(realpath -s test_transformer_ucx.py)
ROOT_DIR=$(dirname $(realpath -s ${0}))
source ${ROOT_DIR}/load_cluster_env.sh

# ${ROOT_DIR}/fornode fuser -k 7777/tcp
${ROOT_DIR}/fornode pkill python

PREV_ADDR=""
CHECKPOINT_PATH=${ROOT_DIR}/checkpoints/ckpt.pt

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
        --check-correctness \
        --checkpoint-path ${CHECKPOINT_PATH} \
    &
  elif [ ${i} == $((N_NODES - 1)) ]; then
    ssh -o StrictHostKeyChecking=no ${NODE_ADDR} \
      ${PYTHON_EXEC} ${PYTHON_SCRIPT} \
        --prev-address ${PREV_ADDR} \
        --prev-port 7777 \
        --rank ${i} \
        --world-size ${N_NODES} \
        --check-correctness \
        --checkpoint-path ${CHECKPOINT_PATH} \
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
        --check-correctness \
        --checkpoint-path ${CHECKPOINT_PATH} \
    &
  fi
  PREV_ADDR=${NODE_ADDR}
done
wait
