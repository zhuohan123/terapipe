# TeraPipe: Token-Level Pipeline Parallelism for Training Large-Scale Language Models

- `test_transformer_terapipe.py`: The entry point for the main pipeline logic and the combination with other parallel training methods.
- `mpirun_terapipe.sh`: Distributed execution with MPI for `test_transformer_terapipe.py`.
- `layer_latency.py`: Collecting Transformer layers' forward and backward latencies for the performance model used by dynamic programming.
- `p2p_comm_latency.py`: Collecting the communication latency for the performance model used by dynamic programming. 
- `dynamic_programming.py`: Train the performance model and perform dynamic programming to find the optimal slicing scheme.
- `transformer_models.py`: Definitions and implementations of the Transformer models.
