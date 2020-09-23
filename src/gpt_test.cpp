#include <chrono>
#include "gpt_local_pipeline.h"

int main() {
  int batch_size = 1;
  int seq_len = 2048;
  int embedding_dim = 3072;
  int n_slices = 16;
  int n_layers = 6;

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0).requires_grad(false);
  torch::Tensor x = torch::randn({seq_len, batch_size, embedding_dim}, options);

  // SingleDeviceGPT gpt_s(12, embedding_dim, embedding_dim / 64, embedding_dim * 4, torch::Device("cuda:0"));
  // auto r = gpt_s.forward(x, {});
  // auto fake_loss = std::get<0>(r).mean();
  // std::cout << fake_loss << std::endl;
  // fake_loss.backward();

  MultiDeviceGPT gpt_m(1, n_layers, embedding_dim, embedding_dim / 64, embedding_dim * 4);
  for (int i = 0; i < 10; i++) {
    auto start_time = std::chrono::high_resolution_clock::now();
    gpt_m.zero_grad();
    torch::Tensor t = gpt_m.forward(x, n_slices);
    auto fake_loss = t.mean();
    std::cout << fake_loss << std::endl;
    fake_loss.backward();
    auto current_time = std::chrono::high_resolution_clock::now();
    std::cout << "Duration = " << std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count()
              << " ms" << std::endl;
  }
  return 0;
}
