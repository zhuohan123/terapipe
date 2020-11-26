# Archive

This folder includes files that are no longer useful for the experiments.

## Single Node Experiments

We implement a set of single node algorithms, including:
1. Running the whole model on a single GPU by:
   ```bash
   MODEL=test # Name of the model to test (see MODEL_CONFIGS)
   N_STEPS=10 # Number of testing steps to run
   python test_transformer_single_node.py \
     --type single \
     --model $MODEL \
     --n-steps $N_STEPS
   ```
2. GPipe (with batch size 1) on all GPUs in the node:
   ```bash
   MODEL=test # Name of the model to test (see MODEL_CONFIGS)
   N_STEPS=10 # Number of testing steps to run
   python test_transformer_single_node.py \
     --type gpipe \
     --model $MODEL \
     --n-steps $N_STEPS
   ```
3. Pipelining on sequence length dimension on all GPUs in the node:
   ```bash
   MODEL=test # Name of the model to test (see MODEL_CONFIGS)
   N_SLICES=8 # Number of input shards (currently we uniformly slice the input)
   N_STEPS=10 # Number of testing steps to run
   python test_transformer_single_node.py \
     --type seqpipe \
     --model $MODEL \
     --n-slices $N_SLICES \
     --n-steps $N_STEPS
   ```

## Multi-Node Pipelining with NCCL
Run multi-node pipelining with `run_nccl.sh`:
```bash
N_NODES=1 # Number of nodes in the cluster
N_GPUS=1 # Number of GPUs per node
MODEL=test # Name of the model to test (see MODEL_CONFIGS)
N_SLICES=8 # Number of input shards (currently we uniformly slice the input)
N_STEPS=10 # Number of testing steps to run
./run_nccl.sh $N_NODES $N_GPUS $MODEL $N_SLICES $N_STEPS
```

## Multi-Node Megatron-LM Baseline
Run multi-node Megatron-LM baseline with `run_megatron.sh`:
```bash
N_NODES=1 # Number of nodes in the cluster
MODEL=test # Name of the model to test (see MODEL_CONFIGS)
N_STEPS=10 # Number of testing steps to run
./run_megatron.sh $N_NODES $MODEL $N_STEPS
```

## Grid Search Forward Pass Running Time for Different Settings
We implement a grid search script to get the runtime of multiple settings. See the code for more details.
```bash
python test_transformer_single_node.py --type gridsearch
python test_transformer_single_node.py --type gridseqlen
```

## Check the correctness of our implementations:
We implement a sanity check for the correctness of our implementations. First, we save the model parameters, input, and corresponding gradients of a single device run:
```bash
MODEL=test # Name of the model to test (see MODEL_CONFIGS)
python test_transformer_single_node.py \
 --type correctness \
 --model $MODEL \
 --checkpoint-path checkpoints/ckpt.pt
```
With this saved checkpoint, we can test the correctness of our implementations:
1. Sanity check with our single device implementation:
   ```bash
   MODEL=test # Same model as the one just saved
   python test_transformer_single_node.py \
    --type single_correctness \
    --model $MODEL \
    --checkpoint-path checkpoints/ckpt.pt
   ```
2. Check the difference of gradients between the single device implementation and pipelining in a single node:
   ```bash
   MODEL=test # Same model as the one just saved
   python test_transformer_single_node.py \
    --type seqpipe_correctness \
    --model $MODEL \
    --checkpoint-path checkpoints/ckpt.pt
   ```
3. Check the difference of gradients between the single device implementation and multi-node pipelining with NCCL:
   ```bash
   N_NODES=1 # Number of nodes in the cluster
   N_GPUS=1 # Number of GPUs per node
   MODEL=test # Same model as the one just saved
   N_SLICES=8 # Number of input shards
   N_STEPS=10 # Number of testing steps to run
   ./check_nccl_correctness.sh $N_NODES $N_GPUS $MODEL $N_SLICES $N_STEPS
   ```
