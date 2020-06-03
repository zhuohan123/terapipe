import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        x, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = y + x
        y = x
        x = self.fc_ln(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = y + x
        return x


if __name__ == "__main__":
    batch_size = 4
    seq_len = 32
    n_layers = 6
    embedding_dim = 128
    ffn_embedding_dim = embedding_dim * 4
    num_attention_heads = 8
    transformer_layers = [
        TransformerLayer(embedding_dim, ffn_embedding_dim, num_attention_heads)
        for _ in range(n_layers)
    ]
    attn_mask = torch.tril(torch.ones(seq_len, seq_len))
    x = torch.randn(seq_len, batch_size, embedding_dim)
    for layer in transformer_layers:
        x = layer(x, attn_mask)
    print(x)
