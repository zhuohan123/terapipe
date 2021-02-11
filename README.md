# Large-Scale Language Modeling with Pipeline Parallelism

In this project, we propose to use pipeline parallelism for large-scale language modeling. Our contributions include:

1. We discover a new dimension (sequence length dimension) for pipeline parallelism on Transformer-based language models. This removes the obstacles to applying previous pipeline parallelism methods on large-scale language models.
2. We show that the optimal size for input shards for pipeline parallelism is only dependent on the compute bound of a single device, while independent with other factors such as the granularity of the pipeline.
3. We systematically analyze the trade-off space between pipeline parallelism and model parallelism based on parallel matrix multiplication. We provide clear guidelines on how to choose between the two algorithms and how to combine them given the heterogeneity of interconnection speeds between different devices.
4. With all proposed algorithms, we greatly accelerate the largest GPT-3 model, without modifying any of the original synchronous training procedure.

## Cluster Setup and Installation

See [cluster/README.md](cluster/README.md) to set up a cluster for developing and testing. After that, clone the repo to the NFS directory `~/efs` shared by all nodes:
```bash
cd ~/efs
git clone https://github.com/zhuohan123/model-parallel-speed-test.git
```

## Model configurations

See `MODEL_CONFIGS` dictionary in [transformer_models.py](transformer_models.py) for the list of the models we are testing on.


## Run all experiments

Pipelining on sequence length dimension on all GPUs in the node:
```bash
# number of nodes, number of gpus per node, model parallel size, 
# pipeline parallel size, model name, number of slices, number of steps
N_NODES=1 # Number of nodes in the cluster
N_GPUS=1 # Number of GPUs per node
MODEL_PARALLEL_SIZE=1 # Number of devices in a single model parallel (parallel matmul) groups
PIPELINE_PARALLEL_SIZE=1 # Number of stages for pipelining. 
# Note that $N_NODES * $N_GPUS == $MODEL_PARALLEL_SIZE * $PIPELINE_PARALLEL_SIZE
MODEL=test # Name of the model to test (see MODEL_CONFIGS)
N_SLICES=8 # Number of input shards (currently we uniformly slice the input)
N_STEPS=10 # Number of testing steps to run
EXTRA_ARGS="--mixed-precision"
./mpirun_terapipe.sh $N_NODES $N_GPUS $MODEL_PARALLEL_SIZE $PIPELINE_PARALLEL_SIZE $MODEL $N_SLICES $N_STEPS $EXTRA_ARGS
```

## Latency Model

### Data collection

Edit `auto_latency_benchmark.sh` and add your model for computation latency evaluation.
Run `./auto_latency_benchmark.sh` over 1 p3.16xlarge machine.
Outputs in `performance_model_data`.

Edit `p2p_comm_latency.py.py` and add your model for communication latency evaluation.
Run `./p2p_comm_latency.sh` over 2 p3.16xlarge machines.
Outputs in `performance_model_data`.

### Fit latency model and generate optimal slices with DP.

Edit and run `latency_model.py` to generate the optimal slices with DP. Results are saved in `dp_results.json`.

### Evaluate the optimal slices.

Edit and run `auto_mpirun_dp_slices_evaluation.sh`. Results under `dp_evaluation_results`.

## Useful scripts:

Get the IPs of all the worker nodes in the cluster:

```bash
python scripts/get_worker_ips.py
```

Load `$MY_IPADDR`, `$OTHERS_IPADDR`, `$ALL_IPADDR` as environment variables:

```bash
source scripts/load_cluster_env.sh
```

Run the same command on all nodes (useful for killing processes and check states):

```bash
scripts/fornode pkill python
scripts/fornode nvidia-smi
```
