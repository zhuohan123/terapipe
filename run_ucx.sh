#!/bin/bash

if [ "$#" -lt 2 ]; then echo "$(tput setaf 1)[ERROR]$(tput sgr 0) number of nodes required"; exit -1; fi
if [ "$#" -gt 2 ]; then echo "$(tput setaf 1)[ERROR]$(tput sgr 0) too many arguments: $#"; exit -1; fi

N_NODES=$1

PYTHON_EXEC=/home/ubuntu/anaconda3/envs/ucx/bin/python
ROOT_DIR=$(dirname $(realpath -s ${0}))
source ${ROOT_DIR}/load_cluster_env.sh

PREV_ADDR=""

for i in $(seq ${N_NODES}); do
  NODE_ADDR=${ALL_IPADDR[i]}
  echo ${i} "=>" ${NODE_ADDR}
  if [ ${i} == 0 ]; then
    ssh -o StrictHostKeyChecking=no ${NODE_ADDR} \
       ${PYTHON_EXEC} test_transformer_ucx.py \
        --my-address ${NODE_ADDR} \
        --my-port 7777 \
    &
  elif [ ${i} == $((N_NODES - 1)) ]; then
    ssh -o StrictHostKeyChecking=no ${NODE_ADDR} \
      ${PYTHON_EXEC} test_transformer_ucx.py \
        --prev-address ${PREV_ADDR} \
        --prev-port 7777 \
    &
  else
    ssh -o StrictHostKeyChecking=no ${NODE_ADDR} \
      ${PYTHON_EXEC} test_transformer_ucx.py \
        --my-address ${NODE_ADDR} \
        --my-port 7777 \
        --prev-address ${PREV_ADDR} \
        --prev-port 7777 \
    &
  fi
  PREV_ADDR=${NODE_ADDR}
done
wait
