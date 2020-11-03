import numpy as np
import argparse
import time
import torch
import torch.distributed as dist
import nccl
from transformer_models import (
    TransformerConfig, MODEL_CONFIGS, uniform_slice_x
)

WARM_UP_ROUNDS = 5


class NCCLTransformerRunner:
    def __init__(self, config, n_slices, distributed_init_method, world_size,
                 rank, local_rank, n_steps):
        self.config = config
        self.n_layers = self.config.n_layers // self.config.n_devices
        self.n_slices = n_slices
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
        )
        self.comm = nccl.get_nccl_communicator(local_rank, rank, world_size)
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.n_steps = n_steps
        print("after initialization")
        # self.layers = self.config.create_layers_gpu()
        # self.all_parameters = []
        # for layer in self.layers:
        #     self.all_parameters += list(layer.parameters())
        # self.n_params = len(self.all_parameters)
        # self.optimizer = torch.optim.SGD(self.all_parameters, lr=1e-10)

    def step(self):
        if self.rank != 0:
            input_x = self.config.create_inputs_empty()
        else:
            input_x = self.config.create_inputs()
        sliced_x = uniform_slice_x(input_x, self.n_slices)

        # forward
        attn_caches = [None] * len(self.layers)
        all_attn_hiddens = [[]]
        all_attn_hiddens_detached = [[]]
        all_inputs = []
        all_outputs = []
        start_time = time.time()
        for i in range(self.n_slices):
            x = sliced_x[i]
            if self.rank > 0:
                self.comm.recv_tensor(x, self.rank - 1)
            x.requires_grad_()
            all_inputs.append(x)
            new_attn_caches_detached = []
            attn_hiddens = []
            attn_hiddens_detached = []
            for layer, attn_cache in zip(self.layers, attn_caches):
                x, new_attn_cache = layer(x, attn_cache)
                attn_hiddens += [v for k, v in new_attn_cache.items()]
                new_attn_cache_detached = {k: v.detach().requires_grad_() for k, v in new_attn_cache.items()}
                attn_hiddens_detached += [v for k, v in new_attn_cache_detached.items()]
                new_attn_caches_detached.append(new_attn_cache_detached)
            attn_caches = new_attn_caches_detached
            all_attn_hiddens.append(attn_hiddens)
            all_attn_hiddens_detached.append(attn_hiddens_detached)
            all_outputs.append(x)
            if self.rank < self.world_size - 1:
                self.comm.send_tensor(x, self.rank + 1)
        print("rank", self.rank, "forward_time", time.time() - start_time, flush=True)

        # backward
        start_time = time.time()
        self.optimizer.zero_grad()

        if self.rank == self.world_size - 1:
            print("rank", self.rank, "calculate loss", flush=True)
            concated_outputs = torch.cat(all_outputs, dim=0)
            loss = torch.mean(concated_outputs)
            grad_all_outputs = torch.autograd.grad(loss, all_outputs)
            print("rank", self.rank, "finish calculating loss", flush=True)

        a = []
        da = []
        if self.rank < self.world_size - 1:
            grad_x = self.config.create_inputs_empty()
            sliced_grad_x = uniform_slice_x(grad_x, self.n_slices)
        for i in reversed(range(self.n_slices)):
            if self.rank == self.world_size - 1:
                dy = grad_all_outputs[i]
            else:
                dy = sliced_grad_x[i]
                self.comm.recv_tensor(dy, self.rank + 1)
            y = all_outputs[i]
            x = all_inputs[i]
            outputs = [y] + a
            grad_outputs = [dy] + da
            inputs = self.all_parameters + [x] + all_attn_hiddens_detached[i]
            all_grads = torch.autograd.grad(outputs, inputs, grad_outputs)
            dw = all_grads[:self.n_params]
            dx = all_grads[self.n_params]
            da = list(all_grads[self.n_params + 1:])
            a = all_attn_hiddens[i]
            if self.rank > 0:
                self.comm.send_tensor(dx, self.rank - 1)
            for grad_w, w in zip(dw, self.all_parameters):
                if w.grad is None:
                    w.grad = grad_w.detach()
                else:
                    w.grad += grad_w
        self.optimizer.step()
        print("rank", self.rank, "backward_time", time.time() - start_time, flush=True)
        torch.cuda.synchronize()

    def run(self):
        all_step_times = []
        for _ in range(self.n_steps):
            start_time = time.time()
            self.step()
            step_time = time.time() - start_time
            all_step_times.append(step_time)
            print("rank", self.rank, "step_time:", step_time, flush=True)
        if len(all_step_times) > WARM_UP_ROUNDS:
            print("rank", self.rank,
                  "step_time_mean:", np.mean(all_step_times[WARM_UP_ROUNDS:]),
                  "step_time_std:", np.std(all_step_times[WARM_UP_ROUNDS:]),
                  flush=True)


def main():
    parser = argparse.ArgumentParser(description='Pipeline + Megatron-LM')
    parser.add_argument('ip_address', type=str, help='the IP address of the head node')
    parser.add_argument('-p', '--port', type=int, help='the port of the head node')
    parser.add_argument('--rank', metavar='I', type=int, default=0)
    parser.add_argument('--local-rank', metavar='I', type=int, default=0)
    parser.add_argument('--world-size', metavar='N', type=int, default=1)
    parser.add_argument('--model-parallel-size', metavar='N', type=int, default=1)
    parser.add_argument('--pipeline-parallel-size', metavar='N', type=int, default=1)
    parser.add_argument('--model', metavar='NAME', type=str, default=None,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--n-slices', metavar='N', type=int, default=8)
    parser.add_argument('--n-steps', metavar='N', type=int, default=10)
    args = parser.parse_args()
    config = TransformerConfig(
        batch_size=1,
        seq_len=1024,
        n_layers=48,
        embedding_dim=2048,
        n_devices=args.world_size,
        model_name=args.model,
    )

    distributed_init_method = f'tcp://{args.ip_address}:{args.port}'
    runner = NCCLTransformerRunner(
        config, args.n_slices, distributed_init_method, args.world_size,
        args.rank, args.local_rank, args.n_steps,
    )
    # runner.run()


if __name__ == "__main__":
    main()
