import argparse
import os
import time
import sys
import gc

from apex import optimizers
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

import mpu
import nccl
from utils import set_random_seed
from transformer_models import (
    TransformerConfig, MODEL_CONFIGS, uniform_slice_x,
    ModelParallelTransformerLayer,
    TransformerLayer
)

WARM_UP_ROUNDS = 5
LOSS_SCALE_FACTOR = 128.0


class NCCLTransformerRunner:
    def __init__(self, config, n_slices, distributed_init_method, world_size,
                 model_parallel_size, pipeline_parallel_size, rank, local_rank,
                 n_steps, mixed_precision=False, use_embedding=False, use_mpi=False):
        self.config = config
        self.use_embedding = use_embedding
        self.n_layers = self.config.n_layers // self.config.n_devices
        self.n_slices = n_slices
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
        )
        dist.all_reduce(torch.zeros(1).cuda())

        # TODO: fix the int(self.use_embedding) usage
        mpu.initialize_model_parallel(model_parallel_size, pipeline_parallel_size, int(self.use_embedding))
        set_random_seed(0)
        mpu.model_parallel_cuda_manual_seed(0)
        self.comm = nccl.get_nccl_communicator(local_rank, rank, world_size, use_mpi)
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.model_parallel_size = model_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size

        self.pipeline_parallel_group_rank = mpu.get_pipeline_parallel_group_rank()
        self.model_parallel_group = mpu.get_model_parallel_group()
        self.model_parallel_src_rank = mpu.get_model_parallel_src_rank()
        self.model_parallel_dst_rank = mpu.get_model_parallel_dst_rank()

        self.model_parallel_next_src_rank = mpu.get_model_parallel_next_src_rank()


        self.model_parallel_prev_dst_rank = mpu.get_model_parallel_prev_dst_rank()

        self.n_steps = n_steps

        self.n_layers = (config.n_layers // pipeline_parallel_size
                         + int(rank - int(self.use_embedding) < config.n_layers % pipeline_parallel_size))
        self.layers = [
            ModelParallelTransformerLayer(
                config.embedding_dim,
                config.ffn_embedding_dim,
                config.num_attention_heads,
                device="cuda",
            )
            for _ in range(self.n_layers)
        ]
        if self.use_embedding and self.rank < int(self.use_embedding):
            embedding_layer, tied_output_layer, _, _ = self.config.create_embedding_and_inputs()
            self.layers = [embedding_layer.cuda(), tied_output_layer.cuda()]

        # ensure layers are destroyed from GPU memory if this is an embedding node
        gc.collect()

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
                p.requires_grad_()

        if self.mixed_precision:
            self.optimizer = optimizers.FusedSGD(self.master_parameters, lr=1e-10)
        else:
            self.optimizer = torch.optim.SGD(self.all_parameters, lr=1e-10)

    def step_embedding(self):
        _, _, input_x, target = self.config.create_embedding_and_inputs()
        input_x, target = input_x.cuda(), target.cuda()

        # embedding -> send to first pipeline group -> wait recv final outputs -> compute loss+embedding backwards pass -> send grad to last pipeline group
        # -> wait recv first pipeline group gradients -> compute embedding backwards pass
        sliced_x_embedding = uniform_slice_x(input_x, self.n_slices)
        sliced_x_embedding_outputs = []

        # TODO: implement multi-GPU embedding group
        start_time = time.time()

        # compute embedding layer and send embedding output to first pipeline group
        for i in range(self.n_slices):
            embedding_output = self.layers[0](sliced_x_embedding[i])
            self.comm.send_tensor(embedding_output, self.model_parallel_next_src_rank)

            sliced_x_embedding_outputs.append(embedding_output)

        final_outputs = uniform_slice_x(self.config.create_inputs_empty(), self.n_slices)
        if self.mixed_precision:
            final_outputs = [o.half() for o in final_outputs]
        final_embedding_outputs = [None for i in range(self.n_slices)]

        # receive final layer output from last pipeline group and compute softmax
        for i in range(self.n_slices):
            self.comm.recv_tensor(final_outputs[i], self.model_parallel_prev_dst_rank)
            final_outputs[i].requires_grad = True

            final_embedding_outputs[i] = self.layers[1](final_outputs[i])

        print("rank", self.rank, "forward_time", time.time() - start_time, flush=True)
        start_time = time.time()
        # compute loss
        # TODO: pipeline this computation
        grad_all_outputs = self.compute_loss(final_embedding_outputs, target)

        embedding_dy_shape = None

        # embedding backwards pass + send dy to last pipeline group
        for i in reversed(range(self.n_slices)):
            dy = grad_all_outputs[i]

            y = final_embedding_outputs[i]
            x = final_outputs[i]
            outputs = [y]
            grad_outputs = [dy]
            inputs = self.all_parameters + [x]
            
            all_grads = torch.autograd.grad(outputs, inputs, grad_outputs)
            dw = all_grads[:self.n_params]
            dx = all_grads[self.n_params]

            self.comm.send_tensor(dx, self.model_parallel_prev_dst_rank)

            embedding_dy_shape = dx.size()
            
            for grad_w, w in zip(dw, self.all_parameters):
                if w.grad is None:
                    w.grad = grad_w.detach()
                else:
                    w.grad += grad_w

        # first embedding layer backwards pass
        embedding_dy = [torch.zeros(embedding_dy_shape).cuda().float() for i in range(self.n_slices)]
        if self.mixed_precision:
            embedding_dy = [o.half() for o in embedding_dy]
        for i in reversed(range(self.n_slices)):
            self.comm.recv_tensor(embedding_dy[i], self.model_parallel_next_src_rank)

            sliced_x_embedding_outputs[i].backward(embedding_dy[i])


        # copy FP16 model gradients to FP32 master before comparing/optimizing
        if self.mixed_precision:
            for model_param, master_param in zip(self.all_parameters, self.master_parameters):
                if master_param.grad is None:
                    master_param.grad = master_param.new(*master_param.size())
                master_param.grad.copy_(model_param.grad)

                # descale master weights
                master_param.grad.mul_(1. / LOSS_SCALE_FACTOR)

        self.optimizer.step()
        if self.mixed_precision:
            # copy master updated FP32 parameters back to FP16
            with torch.no_grad():
                for model_param, master_param in zip(self.all_parameters, self.master_parameters):
                    model_param.copy_(master_param)

        print("rank", self.rank, "backward_time", time.time() - start_time, flush=True)



        


    def compute_loss(self, all_outputs, target=None):
        if self.use_embedding and self.rank >= int(self.use_embedding):
            return None

        if not self.use_embedding and self.pipeline_parallel_group_rank != self.pipeline_parallel_size - 1:
            return None

        concated_outputs = torch.cat(all_outputs, dim=0)

        if self.mixed_precision:
            # cast reductions to FP32
            concated_outputs = concated_outputs.float()

        if self.use_embedding:
            criterion = nn.CrossEntropyLoss()
            concated_outputs = concated_outputs.permute(1, 2, 0)
            target = target.permute(1, 0)
            loss = criterion(concated_outputs, target)
        else:
            loss = torch.mean(concated_outputs)

        # scale up the loss at the source for FP16, then de-scale when each
        # worker performs step() or correctness checks
        if self.mixed_precision:
            loss = loss.float() * LOSS_SCALE_FACTOR
            loss = loss.half()
        print("rank", self.rank, "calculate loss", flush=True)
        grad_all_outputs = torch.autograd.grad(loss, all_outputs)

        
        print("rank", self.rank, "finish calculating loss", flush=True)
        return grad_all_outputs
            

    def step(self):
        # if we're using embedding, we always call the normal create_inputs()
        if self.rank != 0 or self.use_embedding:
            input_x = self.config.create_inputs_empty()
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
            if self.rank == self.model_parallel_src_rank:
                if (not self.use_embedding and self.pipeline_parallel_group_rank > 0) or self.use_embedding:
                    self.comm.recv_tensor(x, self.model_parallel_prev_dst_rank)
            dist.broadcast(x, self.model_parallel_src_rank, group=self.model_parallel_group)
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

            send_cond = self.rank == self.model_parallel_dst_rank
            if not self.use_embedding:
                send_cond = send_cond and self.pipeline_parallel_group_rank < self.pipeline_parallel_size - 1
            if send_cond:
                self.comm.send_tensor(x, self.model_parallel_next_src_rank)

        print("rank", self.rank, "forward_time", time.time() - start_time, flush=True)
        # backward
        start_time = time.time()
        if self.mixed_precision:
            for layer in self.layers:
                layer.zero_grad()
        else:
            self.optimizer.zero_grad()

        grad_all_outputs = self.compute_loss(all_outputs)

        a = []
        da = []
        if self.pipeline_parallel_group_rank < self.pipeline_parallel_size - 1 or self.use_embedding:
            grad_x = self.config.create_inputs_empty()
            if self.mixed_precision:
                grad_x = grad_x.half()
            sliced_grad_x = uniform_slice_x(grad_x, self.n_slices)

        for i in reversed(range(self.n_slices)):
            if self.pipeline_parallel_group_rank == self.pipeline_parallel_size - 1 and not self.use_embedding:
                dy = grad_all_outputs[i]
            else:
                dy = sliced_grad_x[i]
                if self.rank == self.model_parallel_dst_rank:
                    self.comm.recv_tensor(dy, self.model_parallel_next_src_rank)
                dist.broadcast(dy, self.model_parallel_dst_rank, group=self.model_parallel_group)
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

            if self.rank == self.model_parallel_src_rank and (
                (self.pipeline_parallel_group_rank > 0 and not self.use_embedding) or self.use_embedding
            ):
                self.comm.send_tensor(dx, self.model_parallel_prev_dst_rank)
            for grad_w, w in zip(dw, self.all_parameters):
                if w.grad is None:
                    w.grad = grad_w.detach()
                else:
                    w.grad += grad_w
        # copy FP16 model gradients to FP32 master before comparing/optimizing
        if self.mixed_precision:
            for model_param, master_param in zip(self.all_parameters, self.master_parameters):
                if master_param.grad is None:
                    master_param.grad = master_param.new(*master_param.size())
                master_param.grad.copy_(model_param.grad)

                # descale master weights
                master_param.grad.mul_(1. / LOSS_SCALE_FACTOR)

        self.optimizer.step()
        if self.mixed_precision:
            # copy master updated FP32 parameters back to FP16
            with torch.no_grad():
                for model_param, master_param in zip(self.all_parameters, self.master_parameters):
                    model_param.copy_(master_param)

        print("rank", self.rank, "backward_time", time.time() - start_time, flush=True)
        torch.cuda.synchronize()

    def run(self):
        all_step_times = []

        step_func = self.step
        if self.use_embedding and self.rank < int(self.use_embedding):
            step_func = self.step_embedding

        for _ in range(self.n_steps):
            start_time = time.time()
            step_func()
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
    parser.add_argument('--mixed-precision', action='store_true', default=False)
    parser.add_argument('--use-mpi', action='store_true', default=False)
    parser.add_argument('--use-embedding', action='store_true', default=False)

    args = parser.parse_args()
    if args.use_mpi:
        args.world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE'))
        args.rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
        args.local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))

    config = TransformerConfig(
        batch_size=1,
        seq_len=1024,
        n_layers=48,
        embedding_dim=2048,
        n_devices=args.world_size,
        model_name=args.model,
    )

    if args.use_embedding:
        embedding_rank_size = 1
    else:
        embedding_rank_size = 0
    assert args.world_size == args.model_parallel_size * args.pipeline_parallel_size + embedding_rank_size, \
        "Data parallel is not implemented yet"
    distributed_init_method = f'tcp://{args.ip_address}:{args.port}'
    runner = NCCLTransformerRunner(
        config, args.n_slices, distributed_init_method, args.world_size,
        args.model_parallel_size, args.pipeline_parallel_size,
        args.rank, args.local_rank, args.n_steps, mixed_precision=args.mixed_precision,
        use_embedding=args.use_embedding,
        use_mpi=args.use_mpi
    )
    runner.run()
    sys.exit(0)


if __name__ == "__main__":
    main()
