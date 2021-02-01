#!/bin/bash
if [ "$#" -lt 4 ]; then echo "$(tput setaf 1)[ERROR]$(tput sgr 0) number of nodes, number of gpus per node, model name, setting index required"; exit -1; fi
if [ "$#" -gt 4 ]; then echo "$(tput setaf 1)[ERROR]$(tput sgr 0) too many arguments: $#"; exit -1; fi
N_NODES=$1
N_GPUS=$2 # per node
MODEL=$3
INDEX=$4

PYTHON_EXEC=/home/ubuntu/anaconda3/bin/python
PYTHON_SCRIPT=$(realpath -s dp_slices_evaluation.py)
ROOT_DIR=$(dirname $(realpath -s ${0}))
source ${ROOT_DIR}/scripts/load_cluster_env.sh

# ${ROOT_DIR}/scripts/fornode fuser -k 7777/tcp
${ROOT_DIR}/scripts/fornode "pgrep -fl python | awk '!/batch_test\.py/{print $1}' | xargs sudo kill"

echo ALL_IPADDR ${ALL_IPADDR[@]}
all_hosts=$(echo ${ALL_IPADDR[@]:0:$N_NODES} | sed 's/ /,/g')

for p in $(ps aux | grep python | grep 7777 | grep model | grep -v grep | awk '{print $2}'); do kill -9 $p; done

# '--oversubscribe' enables MPI to run muliple processes per node.
/home/ubuntu/anaconda3/bin/mpirun --mca btl_tcp_if_exclude lo,docker0 --mca oob_tcp_if_exclude lo,docker0 \
    --map-by ppr:$N_GPUS:node --oversubscribe -H $all_hosts \
        $PYTHON_EXEC $PYTHON_SCRIPT \
        $MY_IPADDR -p 7777 \
        --model-name $MODEL --index $INDEX
