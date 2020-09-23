#pragma once

#include <thread>
#include "queue.h"
#include "transformer_models.h"

using PacketType = std::pair<int, torch::Tensor>;
using PipelineDataQueue = ConsumerProducerQueue<PacketType>;

struct MultiDeviceGPT : torch::nn::Module {
  MultiDeviceGPT(int n_devices, int n_layers, int embedding_dim, int num_attention_heads, int ffn_embedding_dim);
  torch::Tensor forward(torch::Tensor x, int n_slices);
  void stop();
  ~MultiDeviceGPT() { stop(); }

  int n_devices_;
  std::vector<std::shared_ptr<SingleDeviceGPT>> segments;
  std::vector<PipelineDataQueue> queues;
  std::vector<std::thread> workers;
};
