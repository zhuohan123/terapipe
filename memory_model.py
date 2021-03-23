from transformer_models import MODEL_CONFIGS


def peak_memory_per_gpu(model_name,
                        batch_size,
                        n_nodes,
                        n_data_parallel_replicas=1,
                        gpus_per_megatronlm_shard=8,
                        gpus_per_node=8):
    """Compute peak memory per GPU in GBs.

    Args:
        model_name (str): The name of the model.
        batch_size (int): Batch size per data parallel shard.
        n_nodes (int): Number of nodes per data parallel shard.
        n_data_parallel_replicas (int): Number of data parallel replicas.
        gpus_per_megatronlm_shard (int, optional): Number of GPUs per Megatron-LM shard. Defaults to 8.
        gpus_per_node (int, optional): Number of GPUs per node. Defaults to 8.

    Returns:
        float: Peak memory estimated in GBs.
    """
    n_layers, hidden_size, sequence_length, num_attention_heads = MODEL_CONFIGS[model_name]
    try:
        assert n_layers % n_nodes == 0
        assert gpus_per_node % gpus_per_megatronlm_shard == 0
        assert (n_nodes * gpus_per_node) % n_data_parallel_replicas == 0
    except AssertionError as e:
        print(e, flush=True)
        return float('inf')
    layers_per_megatronlm_shard = n_layers / (n_nodes * gpus_per_node) * (
                gpus_per_megatronlm_shard * n_data_parallel_replicas)

    SIZEOF_FLOAT16 = 2
    SIZEOF_FLOAT32 = 4
    B = batch_size
    L = sequence_length
    H = hidden_size
    C = B * L * H * SIZEOF_FLOAT16
    A = B * num_attention_heads * L * L
    M = gpus_per_megatronlm_shard
    # ~~~~~~~~~~~
    # Activations
    # ~~~~~~~~~~~
    #
    # Self-attention: (2 + 5 / M) C + A / M * (SIZEOF_FLOAT16 + SIZEOF_FLOAT32)
    #   LayerNorm: C
    #   QKV_in_proj: 3C / M
    #   Q_scale: C / M
    #   attn_weights: A / M * SIZEOF_FLOAT16
    #   attn_probs: A / M * SIZEOF_FLOAT32
    #   attn: C / M
    #   output: C
    self_attention_activation_size = C + 3 * C / M + C / M
    self_attention_activation_size += A / M * (SIZEOF_FLOAT16 + SIZEOF_FLOAT32)
    self_attention_activation_size += C / M + C
    # Feed-forward: (2 + 4 / M) C
    #   LayerNorm: C
    #   FC1: 4C / M
    #   FC2: C
    feed_forward_activation_size = C + 4 * C / M + C
    activation_size = self_attention_activation_size + feed_forward_activation_size
    # ~~~~~~~~~~
    # Parameters
    # ~~~~~~~~~~
    #
    # Self-attention:
    #   LayerNorm: 2 * H
    #   in_proj:  (H * H + H) * 3 / M
    #   out_proj: (H * H + H) / M
    # Feed-forward:
    #   LayerNorm: 2 * H
    #   FC1: (H * H + H) * 4 / M
    #   FC2: (H * H * 4 + H) / M
    n_parameters = 2 * H + (H * H + H) * 3 / M + (H * H + H) / M
    n_parameters += 2 * H + (H * H + H) * 4 / M + (H * 4 * H + H) / M
    # 1 FP16, 1 FP32 copy, 2 FP32 for Adam
    parameter_size = n_parameters * (SIZEOF_FLOAT16 + SIZEOF_FLOAT32 + SIZEOF_FLOAT32 * 2)
    max_gradient_size = n_parameters * SIZEOF_FLOAT32
    layer_memory_size = parameter_size + max(activation_size, max_gradient_size)
    peak_memory_per_gpu = 2 * C + layer_memory_size * layers_per_megatronlm_shard
    return peak_memory_per_gpu / 2 ** 30


def memory_stats(model_name, batch_size):
    """Compute total memory.

    Args:
        model_name (str): The name of the model.
        batch_size (int): Batch size per data parallel shard.

    Returns:
        float: Peak memory estimated in GBs.
    """
    n_layers, hidden_size, sequence_length, num_attention_heads = MODEL_CONFIGS[model_name]

    SIZEOF_FLOAT16 = 2
    SIZEOF_FLOAT32 = 4
    B = batch_size
    L = sequence_length
    H = hidden_size
    C = B * L * H * SIZEOF_FLOAT16
    A = B * num_attention_heads * L * L
    M = 1  # gpus_per_megatronlm_shard
    # ~~~~~~~~~~~
    # Activations
    # ~~~~~~~~~~~
    #
    # Self-attention: (2 + 5 / M) C + A / M * (SIZEOF_FLOAT16 + SIZEOF_FLOAT32)
    #   LayerNorm: C
    #   QKV_in_proj: 3C / M
    #   Q_scale: C / M
    #   attn_weights: A / M * SIZEOF_FLOAT16
    #   attn_probs: A / M * SIZEOF_FLOAT32
    #   attn: C / M
    #   output: C
    self_attention_activation_size = C + 3 * C / M + C / M
    self_attention_activation_size += A * (SIZEOF_FLOAT16 + SIZEOF_FLOAT32)
    self_attention_activation_size += C / M + C
    # Feed-forward: (2 + 4 / M) C
    #   LayerNorm: C
    #   FC1: 4C / M
    #   FC2: C
    feed_forward_activation_size = C + 4 * C / M + C
    activation_size = self_attention_activation_size + feed_forward_activation_size

    # ~~~~~~~~~~
    # Parameters
    # ~~~~~~~~~~
    #
    # Self-attention:
    #   LayerNorm: 2 * H
    #   in_proj:  (H * H + H) * 3 / M
    #   out_proj: (H * H + H) / M
    # Feed-forward:
    #   LayerNorm: 2 * H
    #   FC1: (H * H + H) * 4 / M
    #   FC2: (H * H * 4 + H) / M
    n_parameters = 2 * H + (H * H + H) * 3 / M + (H * H + H) / M
    n_parameters += 2 * H + (H * H + H) * 4 / M + (H * 4 * H + H) / M
    # 1 FP16, 1 FP32 copy, 2 FP32 for Adam
    parameter_size = n_parameters * (SIZEOF_FLOAT16 + SIZEOF_FLOAT32 + SIZEOF_FLOAT32 * 2)
    print(f"{model_name}: activation size = {n_layers * activation_size / 2 ** 30} GB, "
          f"parameter size = {parameter_size * n_layers / 2 ** 30} GB")


if __name__ == "__main__":
    print(peak_memory_per_gpu('gpt3-175b', 1, 32))
    print(peak_memory_per_gpu('gpt3-175b', 1, 48))

    memory_stats('gpt3-175b', 1536)
    memory_stats('gpt3-175b', 2)
