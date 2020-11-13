# This file should only be sourced.

MY_IPADDR=$(hostname -i)
OTHERS_IPADDR=()

# The part is does not depend on the Ray runtime.
all_public_ips=$(ray get-worker-ips ~/ray_bootstrap_config.yaml)
for s in $all_public_ips; do
    # if [ ! -f /tmp/$s.ip ]; then
    ssh -o StrictHostKeyChecking=no $s hostname -i > /tmp/$s.ip &
    # fi
done
wait
for s in $all_public_ips; do
    OTHERS_IPADDR+=($(cat /tmp/$s.ip))
done

# # This part depends on Ray runtime. It could be slightly faster.
# OTHERS_IPADDR=($(python $(dirname $(realpath -s ${BASH_SOURCE[0]}))/get_worker_ips.py 2>/dev/null))
ALL_IPADDR=($MY_IPADDR ${OTHERS_IPADDR[@]})
