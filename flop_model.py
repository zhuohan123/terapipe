from transformer_models import MODEL_CONFIGS


def flop_stats(model_name, batch_size):
    """Compute peak memory per GPU in GBs.

    Args:
        model_name (str): The name of the model.
        batch_size (int): Batch size per data parallel shard.

    Returns:
        float: Peak memory estimated in GBs.
    """
    n_layers, hidden_size, sequence_length, num_attention_heads = MODEL_CONFIGS[model_name]

    B = batch_size
    L = sequence_length
    H = hidden_size
    C = B * L * H
    A = B * num_attention_heads * L * L
    # ~~~~~~~~~~~
    # Activations
    # ~~~~~~~~~~~
    #
    # Self-attention:
    #   LayerNorm: C
    #   QKV_in_proj: 3C * H
    #   Q_scale: C
    #   attn_weights: A * (H / num_attention_heads)
    #   attn_probs: A
    #   attn: C * H
    #   output: C
    self_attention_flop = C + 3 * C * H + C
    self_attention_flop += A * (H / num_attention_heads) + A
    self_attention_flop += C * H + C
    # Feed-forward:
    #   LayerNorm: C
    #   FC1: 4C * H + 4 * C
    #   FC2: 4C * H
    ffn_flop = 4 * C * H + 4 * C + 4 * C * H
    addmul_flops = self_attention_flop + ffn_flop
    return (addmul_flops * 6) * n_layers / 10 ** 9


def efficiency_stats(model_name, batch_size, n_gpus, latency, n_slices, n_stages, method_name):
    V100_fp16_flops = 112
    total_flops = V100_fp16_flops * n_gpus * latency

    flop = flop_stats(model_name, batch_size)
    efficiency = (flop / 1000) / total_flops
    print(
        f'[{method_name}] {model_name} efficiency={efficiency:.4f} ({efficiency * V100_fp16_flops:.4f} TFlops of 112 TFlops) '
        f'theoretical-pipeline-efficiency = {n_slices / (n_stages + n_slices - 1):.4f}')


if __name__ == "__main__":
    V100_fp16_flops = 112  # 14 * 2
    # flop = flop_stats('gpt3-175b', 1536)
    # print(flop / 174.6 / 1536 / 2048)
    # print(flop / 1000)

    efficiency_stats('gpt3-175b', 2, 384, 1.160, n_slices=32, n_stages=48, method_name='w/ TeraPipe')
    efficiency_stats('gpt3-175b', 2, 384, 5.822, n_slices=2, n_stages=48, method_name='w/o TeraPipe')

    efficiency_stats('gpt3-13b', 32, 320, 1.328, n_slices=96, n_stages=40, method_name='w/ TeraPipe')
    efficiency_stats('gpt3-13b', 32, 320, 1.863, n_slices=32, n_stages=40, method_name='w/o TeraPipe')

    efficiency_stats('gpt3-1b', 128, 192, 0.913, n_slices=72, n_stages=24, method_name='Both')
