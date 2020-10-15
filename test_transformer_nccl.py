#!/usr/bin/env python
import numpy as np
import argparse
import traceback
import time
import torch
from transformer_models import (
    TransformerConfig, load_layers, load_grads, load_inputs, MODEL_CONFIGS
)

import os
import sys

# os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_SOCKET_NTHREADS'] = '4'
os.environ['NCCL_NSOCKS_PERTHREAD'] = '4'

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "nccl"))
import py_nccl_sendrecv


def uniform_slice_x(x, n_slices):
    seq_len = x.size()[0]
    sliced_x = []
    start_index = 0
    for i in range(n_slices):
        seq_len_slice = seq_len // n_slices + int(i < seq_len % n_slices)
        sliced_x.append(x[start_index:start_index + seq_len_slice])
        start_index += seq_len_slice
    assert start_index == seq_len
    return sliced_x


WARM_UP_ROUNDS = 5


class NCCLTransformerRunner:
    def __init__(self, config, n_slices, nccl_uniq_id, world_size, rank, local_rank, n_steps,
                 check_correctness=False, checkpoint_path=None):
        self.config = config
        self.n_layers = self.config.n_layers // self.config.n_devices
        self.n_slices = n_slices
        torch.cuda.set_device(local_rank)
        self.comm = py_nccl_sendrecv.NCCL(nccl_uniq_id, world_size)
        self.comm.init_rank(local_rank, rank)
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.n_steps = n_steps
        self.check_correctness = check_correctness
        self.prefix = checkpoint_path
        self.layers = self.config.create_layers_gpu()
        if self.check_correctness:
            load_layers(self.layers,
                        range(self.rank * self.n_layers,
                              self.rank * self.n_layers + self.n_layers),
                        self.prefix)
            print("Rank {} loaded layers: {}-{}".format(
                self.rank, self.rank * self.n_layers,
                self.rank * self.n_layers + self.n_layers))
        self.all_parameters = []
        for layer in self.layers:
            self.all_parameters += list(layer.parameters())
        self.n_params = len(self.all_parameters)
        self.optimizer = torch.optim.SGD(self.all_parameters, lr=1e-10)

    def step(self):
        if self.rank != 0:
            input_x = self.config.create_inputs_empty()
        elif self.check_correctness:
            input_x = load_inputs(self.prefix)
            print("Rank {} loaded input x".format(self.rank))
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
        if self.check_correctness:
            all_ref_grads = load_grads(range(self.rank * self.n_layers,
                                             self.rank * self.n_layers + self.n_layers),
                                       self.prefix)
            for layer, ref_grads in zip(self.layers, all_ref_grads):
                for param, ref_grad in zip(layer.parameters(), ref_grads):
                    assert param.grad.size() == ref_grad.size()
                    print(torch.mean(torch.abs(param.grad - ref_grad.to(param.grad))))
        else:
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
    parser = argparse.ArgumentParser(description='NCCL based transformer')
    parser.add_argument('--rank', metavar='I', type=int, default=0)
    parser.add_argument('--local-rank', metavar='I', type=int, default=0)
    parser.add_argument('--world-size', metavar='N', type=int, default=1)
    parser.add_argument('--check-correctness', action='store_true')
    parser.add_argument('--checkpoint-path', metavar='PATH', type=str, default=None)
    parser.add_argument('--model', metavar='NAME', type=str, default=None,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--n-slices', metavar='N', type=int, default=8)
    parser.add_argument('--n-steps', metavar='N', type=int, default=10)
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    config = TransformerConfig(
        batch_size=1,
        seq_len=1024,
        n_layers=48,
        embedding_dim=2048,
        n_devices=args.world_size,
        model_name=args.model,
    )

    id_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "nccl_uniq_id")

    if args.rank == 0:
        nccl_uniq_id = py_nccl_sendrecv.get_unique_id()
        with open(id_file, "wb") as f:
            f.write(nccl_uniq_id)
    else:
        time.sleep(3)
        with open(id_file, "rb") as f:
            nccl_uniq_id = f.read()

    runner = NCCLTransformerRunner(
        config, args.n_slices, nccl_uniq_id, args.world_size, args.rank, args.local_rank, args.n_steps,
        check_correctness=args.check_correctness, checkpoint_path=args.checkpoint_path,
    )
    runner.run()


if __name__ == "__main__":
    main()
