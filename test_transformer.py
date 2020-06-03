import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mpu


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # mpu.model_parallel_cuda_manual_seed(seed)


class TransformerLayer(nn.Module):
    def __init__(self, embedding_dim, ffn_embedding_dim, num_attention_heads):
        super().__init__()
        self.attn_ln = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(embedding_dim, num_attention_heads)
        self.fc_ln = nn.LayerNorm(embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)

    def forward(self, x, attn_mask):
        y = x
        x = self.attn_ln(x)
        x, attn_weight = self.attn(x, x, x, attn_mask=attn_mask)
        x = y + x
        y = x
        x = self.fc_ln(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = y + x
        return x


class SingleDeviceTransformer(nn.Module):
    def __init__(self, transformer_layers, seq_len):
        super().__init__()
        self.layers = nn.ModuleList(transformer_layers)
        self.seq_len = seq_len
        self.register_buffer("attn_mask", torch.triu(torch.full((seq_len, seq_len), -1e10), 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x, self.attn_mask)
        return x


def main():
    set_random_seed(0)
    batch_size = 1
    seq_len = 32
    n_layers = 6
    embedding_dim = 128
    ffn_embedding_dim = embedding_dim * 4
    num_attention_heads = 8
    transformer_layers = [
        TransformerLayer(embedding_dim, ffn_embedding_dim, num_attention_heads)
        for _ in range(n_layers)
    ]
    single_device_transformer = SingleDeviceTransformer(transformer_layers, seq_len)
    x = torch.randn(seq_len, batch_size, embedding_dim)
    y = single_device_transformer(x)


if __name__ == "__main__":
    main()
