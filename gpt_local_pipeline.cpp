#include "gpt_local_pipeline.h"

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
    } break; // comfort some compilers
    default:
      // TODO: fail the program here
      std::cerr << "Unknown message type: " << packet.first << std::endl;
      break;
    }
  }
}

MultiDeviceGPT::MultiDeviceGPT(int n_devices, int n_layers, int embedding_dim, int num_attention_heads, int ffn_embedding_dim)
    : n_devices_(n_devices), queues(n_devices + 1), workers(n_devices) {
  for (int i = 0; i < n_devices; i++) {
    int segment_size = n_layers / n_devices + int(i < n_layers % n_devices);
    auto gpt_segment = register_module("segment_" + std::to_string(i),
                                        std::make_shared<SingleDeviceGPT>(segment_size, embedding_dim,
                                                                          num_attention_heads, ffn_embedding_dim,
                                                                          torch::Device("cuda:" + std::to_string(i))));
    workers[i] = std::thread(local_pipeline_stage, gpt_segment, std::ref(queues[i]), std::ref(queues[i + 1]),
                              torch::Device("cuda:" + std::to_string(i)));
    segments.push_back(std::move(gpt_segment));
  }
}

torch::Tensor MultiDeviceGPT::forward(torch::Tensor x, int n_slices) {
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
  PacketType r;
  queues.back().consume(r); // skip the initial message
  for (int i = 0; i < n_slices; i++) {
    queues.back().consume(r);
    results.push_back(r.second);
  }
  return torch::cat(results, 0);
}

void MultiDeviceGPT::stop() {
  queues[0].add(std::make_pair(2, torch::Tensor()));
  for (auto &w : workers) {
    w.join();
  }
}
