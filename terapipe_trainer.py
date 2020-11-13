from apex import optimizers
import torch

from transformer_models import TransformerLayer

LOSS_SCALE_FACTOR = 128.0


class GPTTrainer:
    def __init__(self,
                 transformer_layer_cls: TransformerLayer,
                 n_layers,
                 embedding_dim,
                 ffn_embedding_dim,
                 num_attention_heads,
                 mixed_precision):
        self.layers = [
            transformer_layer_cls(
                embedding_dim, ffn_embedding_dim, num_attention_heads, device="cuda")
                for _ in range(n_layers)
        ]

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
        
        # intermediate states
        self.all_attn_hiddens = [[]]
        self.all_attn_hiddens_detached = [[]]
        self.all_inputs = []
        self.all_outputs = []
        self.n_slices = None

    def forward_corotine(self, n_slices):
        self.n_slices = n_slices
        attn_caches = [None] * len(self.layers)
        for i in range(n_slices):
            x = (yield i)
            x.requires_grad_()
            self.all_inputs.append(x)
            new_attn_caches_detached = []
            attn_hiddens = []
            attn_hiddens_detached = []
            for layer, attn_cache in zip(self.layers, attn_caches):
                x, new_attn_cache = layer(x, attn_cache)
                attn_hiddens.extend(new_attn_cache.values())
                new_attn_cache_detached = {k: v.detach().requires_grad_() for k, v in new_attn_cache.items()}
                attn_hiddens_detached.extend(new_attn_cache_detached.values())
                new_attn_caches_detached.append(new_attn_cache_detached)
            attn_caches = new_attn_caches_detached
            self.all_attn_hiddens.append(attn_hiddens)
            self.all_attn_hiddens_detached.append(attn_hiddens_detached)
            self.all_outputs.append(x)
            yield x
    
    def compute_loss(self):
        concated_outputs = torch.cat(self.all_outputs, dim=0)
        if self.mixed_precision:
            # cast reductions to FP32
            concated_outputs = concated_outputs.float()
        loss = torch.mean(concated_outputs)

        # scale up the loss at the source for FP16, then de-scale when each
        # worker performs step() or correctness checks
        if self.mixed_precision:
            loss = loss.float() * LOSS_SCALE_FACTOR
            loss = loss.half()
        grad_all_outputs = torch.autograd.grad(loss, self.all_outputs)
        return loss.item(), grad_all_outputs

    def zero_grad(self):
        if self.mixed_precision:
            for layer in self.layers:
                layer.zero_grad()
        else:
            self.optimizer.zero_grad()
    
    def backward_coroutine(self, n_slices):
        assert n_slices == self.n_slices
        a = []
        da = []
        for i in reversed(range(n_slices)): 
            dy = (yield i)
            y = self.all_outputs[i]
            x = self.all_inputs[i]
            outputs = [y] + a
            grad_outputs = [dy] + da
            inputs = self.all_parameters + [x] + self.all_attn_hiddens_detached[i]
            all_grads = torch.autograd.grad(outputs, inputs, grad_outputs)
            dx = all_grads[self.n_params]
            yield dx
            da = list(all_grads[self.n_params + 1:])
            a = self.all_attn_hiddens[i]
            # accumulate gradients
            dw = all_grads[:self.n_params]
            for grad_w, w in zip(dw, self.all_parameters):
                if w.grad is None:
                    w.grad = grad_w.detach()
                else:
                    w.grad += grad_w

    def update(self):
        self.all_attn_hiddens = [[]]
        self.all_attn_hiddens_detached = [[]]
        self.all_inputs = []
        self.all_outputs = []
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
