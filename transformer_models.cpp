#include <cmath>
#include <cstdint>
#include <thread>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

#include "queue.h"

#define NEG_INF -1e10
using torch::Tensor;

struct AttentionCache {
  AttentionCache(const Tensor &key, const Tensor &value) : k(key), v(value) {}
  torch::Tensor k, v;
};

using AttentionResultTuple = std::tuple<torch::Tensor, std::unique_ptr<AttentionCache>>;
using SingleDeviceAttentionTuple = std::tuple<torch::Tensor, std::vector<std::unique_ptr<AttentionCache>>>;
using PacketType = std::pair<int, torch::Tensor>;
using PipelineDataQueue = ConsumerProducerQueue<PacketType>;

struct MultiheadLMAttentionWithCache : torch::nn::Module {
  MultiheadLMAttentionWithCache(int embed_dim, int num_heads, const torch::Device &device)
      : embed_dim_(embed_dim), num_heads_(num_heads) {
    in_proj = register_module("in_proj", torch::nn::Linear(embed_dim, embed_dim * 3));
    out_proj = register_module("out_proj", torch::nn::Linear(embed_dim, embed_dim));
    head_dim_ = embed_dim / num_heads;
    assert(embed_dim % num_heads == 0);
    scaling_ = 1.0 / sqrt(head_dim_);
  }

  AttentionResultTuple forward(torch::Tensor x, std::unique_ptr<AttentionCache> cache) {
    int tgt_len = x.size(0);
    int bsz = x.size(1);
    int embed_dim = x.size(2);
    assert(embed_dim == embed_dim_);

    std::vector<Tensor> qkv =
        in_proj->forward(x).contiguous().view({tgt_len, bsz * num_heads_, head_dim_ * 3}).transpose(0, 1).chunk(3, -1);
    Tensor &q = qkv[0];
    Tensor &k = qkv[1];
    Tensor &v = qkv[2];

    q *= scaling_;
    Tensor attn_mask = x.new_full({tgt_len, tgt_len}, NEG_INF).triu(1);
    int src_len = tgt_len;
    if (cache) {
      int cache_len = (cache->k).size(1);
      k = torch::cat({cache->k, k}, 1);
      v = torch::cat({cache->v, v}, 1);
      attn_mask = torch::cat({x.new_zeros({tgt_len, cache_len}), attn_mask}, 1);
      src_len += cache_len;
    }

    auto new_cache = std::make_unique<AttentionCache>(k, v);
    Tensor attn_weights = torch::bmm(q, k.transpose(1, 2));
    assert(attn_weights.size(0) == bsz * num_heads_);
    assert(attn_weights.size(1) == tgt_len);
    assert(attn_weights.size(2) == src_len);
    attn_weights += attn_mask.index({torch::indexing::None, torch::indexing::Slice(), torch::indexing::Slice()});
    Tensor attn_probs = torch::softmax(attn_weights, -1, torch::kFloat32).type_as(attn_weights);
    Tensor attn = torch::bmm(attn_probs, v);
    attn = attn.transpose(0, 1).contiguous().view({tgt_len, bsz, -1});
    attn = out_proj->forward(attn);

    return std::make_tuple(attn, std::move(new_cache));
  }

  torch::nn::Linear in_proj{nullptr}, out_proj{nullptr};
  int embed_dim_;
  int num_heads_;
  int head_dim_;
  double scaling_;
};

struct TransformerLayer : torch::nn::Module {
  TransformerLayer(int embedding_dim, int num_attention_heads, int ffn_embedding_dim, const torch::Device &device) {
    attn_ln = register_module("attn_ln", torch::nn::LayerNorm(std::vector<int64_t>({embedding_dim})));
    attn = register_module("attn",
                           std::make_shared<MultiheadLMAttentionWithCache>(embedding_dim, num_attention_heads, device));
    fc_ln = register_module("fc_ln", torch::nn::LayerNorm(std::vector<int64_t>({embedding_dim})));
    fc1 = register_module("fc1", torch::nn::Linear(embedding_dim, ffn_embedding_dim));
    fc2 = register_module("fc2", torch::nn::Linear(ffn_embedding_dim, embedding_dim));

    attn_ln->to(device);
    attn->to(device);
    fc_ln->to(device);
    fc1->to(device);
    fc2->to(device);
  }

  AttentionResultTuple forward(torch::Tensor x, std::unique_ptr<AttentionCache> attn_cache) {
    int64_t seq_len = x.size(0);
    int64_t batch_size = x.size(1);
    torch::Tensor y = x;
    x = attn_ln->forward(x);
    AttentionResultTuple att = attn->forward(x, std::move(attn_cache));
    x = std::get<0>(att);
    x = y + x;
    y = x;
    x = fc_ln->forward(x);
    x = fc1->forward(x);
    x = torch::relu(x);
    x = fc2->forward(x);
    x = y + x;
    return std::make_tuple(x, std::move(std::get<1>(att)));
  }

  // Use one of many "standard library" modules.
  std::shared_ptr<MultiheadLMAttentionWithCache> attn{nullptr};
  torch::nn::LayerNorm attn_ln{nullptr}, fc_ln{nullptr};
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

struct SingleDeviceGPT : torch::nn::Module {
  SingleDeviceGPT(int n_layers, int embedding_dim, int num_attention_heads, int ffn_embedding_dim,
                  const torch::Device &device) {
    register_module("layers", layers);
    for (int i = 0; i < n_layers; i++) {
      layers->push_back(TransformerLayer(embedding_dim, num_attention_heads, ffn_embedding_dim, device));
    }
  }

  SingleDeviceAttentionTuple forward(torch::Tensor x, std::vector<std::unique_ptr<AttentionCache>> &&attn_caches) {
    std::vector<std::unique_ptr<AttentionCache>> new_attn_caches;
    int size = layers->size();
    if (attn_caches.empty()) {
      attn_caches.resize(size);
    }
    for (int i = 0; i < size; i++) {
      auto result = layers[i]->as<TransformerLayer>()->forward(x, std::move(attn_caches[i]));
      x = std::get<0>(result);
      new_attn_caches.push_back(std::move(std::get<1>(result)));
    }
    return std::make_tuple(x, std::move(new_attn_caches));
  }

  torch::nn::ModuleList layers;
};

void local_pipeline_stage(std::shared_ptr<SingleDeviceGPT> model, PipelineDataQueue &in_queue,
                          PipelineDataQueue &out_queue, torch::Device device) {
  std::vector<std::unique_ptr<AttentionCache>> caches;
  c10::cuda::CUDAGuard g(device);
  while (true) {
    PacketType packet;
    in_queue.consume(packet);
    switch (packet.first) {
    case 0: {
      // received the signal for new batch. pass it down to the pipeline and clear local caches.
      caches.clear();
      out_queue.add(packet);
    } break;
    case 1: {
      torch::Tensor x = packet.second.to(device, torch::kFloat32, /*non_blocking=*/true);
      auto output = model->forward(x, std::move(caches));
      caches = std::move(std::get<1>(output));
      out_queue.add(std::make_pair(1, std::get<0>(output)));
    } break;
    case 2: {
      // received the stop signal. pass it down to the pipeline and return.
      out_queue.add(packet);
      return;
    } break;  // comfort some compilers
    default:
      // TODO: fail the program here
      std::cerr << "Unknown message type: " << packet.first << std::endl;
      break;
    }
  }
}

struct MultiDeviceGPT : torch::nn::Module {
  MultiDeviceGPT(int n_devices, int n_layers, int embedding_dim, int num_attention_heads, int ffn_embedding_dim)
      : n_devices_(n_devices), queues(n_devices + 1) {
    for (int i = 0; i < n_devices; i++) {
      int segment_size = n_layers / n_devices + int(i < n_layers % n_devices);
      auto gpt_segment = register_module("segment_" + std::to_string(i),
                                         std::make_shared<SingleDeviceGPT>(segment_size, embedding_dim,
                                                                           num_attention_heads, ffn_embedding_dim,
                                                                           torch::Device("cuda:" + std::to_string(i))));
      segments.push_back(std::move(gpt_segment));
      workers[i] = std::thread(local_pipeline_stage, gpt_segment, std::ref(queues[i]), std::ref(queues[i + 1]),
                               torch::Device("cuda:" + std::to_string(i)));
    }
  }

  torch::Tensor forward(torch::Tensor x, int n_slices) {
    queues[0].add(std::make_pair(0, torch::Tensor()));
    int seq_len = x.size(0);
    int pos = 0;
    for (int i = 0; i < n_slices; i++) {
      int subseq_len = seq_len / n_slices + int(i < seq_len % n_slices);
      torch::Tensor slice = x.index({torch::indexing::Slice(pos, pos + subseq_len)});
      queues[0].add(std::make_pair(1, slice));
      pos += subseq_len;
    }
    std::vector<torch::Tensor> results;
    for (int i = 0; i < n_slices; i++) {
      PacketType r;
      queues.back().consume(r);
      results.push_back(r.second);
    }
    return torch::cat(results, 0);
  }

  void stop() {
    queues[0].add(std::make_pair(2, torch::Tensor()));
    for (auto& w: workers) {
      w.join();
    }
  }

  ~MultiDeviceGPT() {
    stop();
  }

  int n_devices_;
  std::vector<std::shared_ptr<SingleDeviceGPT>> segments;
  std::vector<PipelineDataQueue> queues;
  std::vector<std::thread> workers;
};

int main() {
  int batch_size = 1;
  int seq_len = 1024;
  int embedding_dim = 512;

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0).requires_grad(false);
  torch::Tensor x = torch::randn({seq_len, batch_size, embedding_dim}, options);

  SingleDeviceGPT gpt_s(12, embedding_dim, 64, 3072 * 2, torch::Device("cuda:0"));
  auto r = gpt_s.forward(x, {});
  auto fake_loss = std::get<0>(r).mean();
  std::cout << fake_loss << std::endl;
  fake_loss.backward();
  return 0;
}
