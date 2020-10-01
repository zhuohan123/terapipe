#!/bin/bash
MY_IPADDR=$(hostname -i)
OTHERS_IPADDR=($(python get_worker_ips.py 2>/dev/null))
ALL_IPADDR=($MY_IPADDR ${OTHERS_IPADDR[@]})
# all_nodes=(${ALL_IPADDR[@]:0:$world_size})
all_hosts=$(echo ${ALL_IPADDR[@]} | sed 's/ /,/g')
mpirun --mca btl_tcp_if_include ens3 --map-by ppr:1:node -H $all_hosts $@
