#pragma once

#include <cstdint>
#include <torch/torch.h>

#define NEG_INF -1e10
using torch::Tensor;

struct AttentionCache {
  AttentionCache(const Tensor &key, const Tensor &value) : k(key), v(value) {}
  torch::Tensor k, v;
};

using AttentionResultTuple = std::tuple<torch::Tensor, std::unique_ptr<AttentionCache>>;
using SingleDeviceAttentionTuple = std::tuple<torch::Tensor, std::vector<std::unique_ptr<AttentionCache>>>;

struct MultiheadLMAttentionWithCache : torch::nn::Module {
  MultiheadLMAttentionWithCache(int embed_dim, int num_heads, const torch::Device &device);
  AttentionResultTuple forward(torch::Tensor x, std::unique_ptr<AttentionCache> cache);

  torch::nn::Linear in_proj{nullptr}, out_proj{nullptr};
  int embed_dim_;
  int num_heads_;
  int head_dim_;
  double scaling_;
};

struct TransformerLayer : torch::nn::Module {
  TransformerLayer(int embedding_dim, int num_attention_heads, int ffn_embedding_dim, const torch::Device &device);
  AttentionResultTuple forward(torch::Tensor x, std::unique_ptr<AttentionCache> attn_cache);

  std::shared_ptr<MultiheadLMAttentionWithCache> attn{nullptr};
  torch::nn::LayerNorm attn_ln{nullptr}, fc_ln{nullptr};
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

struct SingleDeviceGPT : torch::nn::Module {
  SingleDeviceGPT(int n_layers, int embedding_dim, int num_attention_heads, int ffn_embedding_dim,
                  const torch::Device &device);
  SingleDeviceAttentionTuple forward(torch::Tensor x, std::vector<std::unique_ptr<AttentionCache>> &&attn_caches);

  torch::nn::ModuleList layers;
};
