import numpy as np
import argparse
import time
import torch
from transformer_models import (
    TransformerConfig, load_layers, load_grads, load_inputs, MODEL_CONFIGS, uniform_slice_x
)

import os
import sys
from apex import optimizers

# os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_SOCKET_NTHREADS'] = '4'
os.environ['NCCL_NSOCKS_PERTHREAD'] = '4'

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "nccl"))
import py_nccl_sendrecv

WARM_UP_ROUNDS = 5
LOSS_SCALE_FACTOR = 128.0

class NCCLTransformerRunner:
    def __init__(self, config, n_slices, nccl_uniq_id, world_size, rank, local_rank, n_steps,
                 check_correctness=False, checkpoint_path=None, mixed_precision=False):
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

        self.mixed_precision = mixed_precision

        if self.mixed_precision:
            for i in range(len(self.layers)):
                self.layers[i] = self.layers[i].half()

            self.all_parameters = []
            for layer in self.layers:
                self.all_parameters += list(layer.parameters())

            self.master_parameters = [p.clone().detach().float() for p in self.all_parameters]
            for p in self.master_parameters:
                p.requires_grad = True


        if self.mixed_precision:
            self.optimizer = optimizers.FusedSGD(self.master_parameters, lr=1e-10)
        else:
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

        if self.mixed_precision:
            sliced_x = [x.half() for x in sliced_x]

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

        if self.mixed_precision:
            for layer in self.layers:
                layer.zero_grad()
        else:
            self.optimizer.zero_grad()

        if self.rank == self.world_size - 1:
            print("rank", self.rank, "calculate loss", flush=True)
            concated_outputs = torch.cat(all_outputs, dim=0)
            if self.mixed_precision:
                # cast reductions to FP32
                concated_outputs = concated_outputs.float()
            loss = torch.mean(concated_outputs)

            # scale up the loss at the source for FP16, then de-scale when each
            # worker performs step() or correctness checks
            if self.mixed_precision:
                loss = loss.float() * LOSS_SCALE_FACTOR
                loss = loss.half()

            grad_all_outputs = torch.autograd.grad(loss, all_outputs)
            print("rank", self.rank, "finish calculating loss", flush=True)

        a = []
        da = []
        if self.rank < self.world_size - 1:
            grad_x = self.config.create_inputs_empty()
            sliced_grad_x = uniform_slice_x(grad_x, self.n_slices)

            if self.mixed_precision:
                grad_x = grad_x.half()
                sliced_grad_x = sliced_grad_x.half()

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
        # copy FP16 model gradients to FP32 master before comparing/optimizing
        if self.mixed_precision:
            for model_param, master_param in zip(self.all_parameters, self.master_parameters):
                if master_param.grad is None:
                    master_param.grad = torch.autograd.Variable(master_param.data.new(*master_param.data.size()))
                master_param.grad.data.copy_(model_param.grad.data)

                # descale master weights
                master_param.grad.data.mul_(1. / LOSS_SCALE_FACTOR)

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
        if self.mixed_precision:
            # copy master updated FP32 parameters back to FP16
            for model_param, master_param in zip(self.all_parameters, self.master_parameters):
                model_param.data.copy_(master_param.data)
        
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
    parser.add_argument('--mixed-precision', action='store_true', default=False)
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

    # NOTE: we must save id file to a shared filesystem like AWS efs!
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
        mixed_precision=args.mixed_precision
    )
    runner.run()


if __name__ == "__main__":
    main()
