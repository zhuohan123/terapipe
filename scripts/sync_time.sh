#!/bin/bash

sudo apt install -y chrony
sudo sed -i 's/^# information about usuable directives./server 169.254.169.123 prefer iburst minpoll 4 maxpoll 4\n/g' /etc/chrony/chrony.conf
sudo /etc/init.d/chrony restart

if [ "$#" -eq 0 ]; then
    ROOT_DIR=$(dirname $(realpath -s $0))
    source $ROOT_DIR/load_cluster_env.sh
    for node in ${OTHERS_IPADDR[@]}
    do
        ssh -t -t $node "$(realpath -s $0) 0" &
    done
    wait
fi
