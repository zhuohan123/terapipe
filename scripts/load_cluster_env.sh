# This file should only be sourced.

MY_IPADDR=$(hostname -i)
# OTHERS_IPADDR=()
# for s in $(ray get-worker-ips ~/ray_bootstrap_config.yaml); do
#     OTHERS_IPADDR+=($(ssh -o StrictHostKeyChecking=no $s hostname -i));
# done
OTHERS_IPADDR=($(python $(dirname $(realpath -s ${BASH_SOURCE[0]}))/get_worker_ips.py 2>/dev/null))
ALL_IPADDR=($MY_IPADDR ${OTHERS_IPADDR[@]})
