#pragma once

#include "queue.h"
#include "transformer_models.h"
#include <thread>

using PacketType = std::pair<int, torch::Tensor>;
using PipelineDataQueue = ConsumerProducerQueue<PacketType>;

struct MultiDeviceGPT : torch::nn::Module {
  MultiDeviceGPT(int n_devices, int n_layers, int embedding_dim, int num_attention_heads, int ffn_embedding_dim);
  torch::Tensor forward(torch::Tensor x, int n_slices);

  void stop();

  ~MultiDeviceGPT() { stop(); }

  void push_empty_cache_signal() { push(0, torch::Tensor()); }

  void push_stop_signal() { push(2, torch::Tensor()); }

  void push_input_batch(const torch::Tensor &tensor) { push(1, tensor); }

  void push(int msg, const torch::Tensor &tensor) { queues[0].add(std::make_pair(msg, tensor)); }

  void pop(int *msg, torch::Tensor *tensor) {
    PacketType r;
    queues.back().consume(r); // skip the initial message
    *msg = r.first;
    *tensor = r.second;
  }

  int n_devices_;
  std::vector<std::shared_ptr<SingleDeviceGPT>> segments;
  std::vector<PipelineDataQueue> queues;
  std::vector<std::thread> workers;
};
