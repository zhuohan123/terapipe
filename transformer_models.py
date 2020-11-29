from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import queue
import mpu


class AttentionCache(namedtuple('attention_cache', ['k', 'v'])):
    def detach(self):
        return AttentionCache(
            k=self.k.detach().requires_grad_(),
            v=self.v.detach().requires_grad_())

    def __len__(self):
        return int(self.k.size(1))


# https://github.com/NVIDIA/apex/issues/93
# since the purpose of mask is to set exp(val) to 0
# a large negative value serves the same purpose here
NEG_INF = -65504
MODEL_CONFIGS = {
    # n_layers, hidden_size, sequence_length, num_attention_heads
    "test":        (24,   256,  128,   256 // 64),
    "test-l512":   (512,  256,  128,   256 // 64),
    "test-s1024":  (48,  2048, 1024,  2048 // 64),
    "test-s2048":  (48,  2048, 2048,  2048 // 64),
    "test-h3072":  (48,  3072, 2048,  3072 // 64),
    "gpt2-1hm":    (12,   768, 1024,   768 // 64),
    "gpt2-3hm":    (24,  1024, 1024,  1024 // 64),
    "gpt2-7hm":    (36,  1280, 1024,  1280 // 64),
    "gpt2-1b":     (48,  1600, 1024,  1600 // 64),
    "megatron-1b": (40,  1536, 1024,  1536 // 96),
    "megatron-2b": (54,  1920, 1024,  1920 // 96),
    "megatron-4b": (64,  2304, 1024,  2304 // 96),
    "megatron-8b": (72,  3072, 1024,  3072 // 96),
    "gpt3-1hm":    (12,   768, 2048,   768 // 64),
    "gpt3-3hm":    (24,  1024, 2048,  1024 // 64),
    "gpt3-7hm":    (24,  1536, 2048,  1536 // 96),
    "gpt3-1b":     (24,  2048, 2048,  2048 // 128),
    "gpt3-2b":     (32,  2560, 2048,  2560 // 80),
    "gpt3-6b":     (32,  4096, 2048,  4096 // 128),
    "gpt3-13b":    (40,  5120, 2048,  5120 // 128),
    "gpt3-175b":   (96, 12288, 2048, 12288 // 128),
    # This is the model that Megatron-LM can run on
    # 48*8 NVIDIA-V100(16 GB) GPUs without OOM.
    "gpt3-175b-megatron":   (48, 12288//2, 2048, 384),
    # these config are tuned to be fit into a single GPU (NVIDIA V100 16G)
    "gpt2-1hm-single-device":    (10,  768, 2048,  768 // 64),
    "gpt2-3hm-single-device":    (8, 1024, 2048, 1024 // 64),
    "gpt2-7hm-single-device":    (6, 1280, 2048, 1280 // 64),
    "gpt2-1b-single-device":     (4, 1600, 2048, 1600 // 64),
    "gpt3-1b-single-device":     (5, 2048, 2048, 2048 // 128),
    "gpt3-2b-single-device":     (3,  2560, 2048, 2560 // 80),
    "gpt3-6b-single-device":     (2,  4096, 2048, 4096 // 128),
    "gpt3-13b-single-device":    (1,  5120, 2048, 5120 // 128),
    # this is for single node
    "gpt3-175b-single-node":   (2, 12288, 2048, 12288 // 128),
}


class TransformerConfig:
    def __init__(
        self,
        batch_size=1,
        seq_len=1024,
        n_layers=12,
        embedding_dim=768,
        ffn_embedding_dim=None,
        num_attention_heads=None,
        placement_orders=None,
        n_devices=None,
        model_name=None,
    ):
        self.batch_size = batch_size
        if model_name is None:
            self.seq_len = seq_len
            self.n_layers = n_layers
            self.embedding_dim = embedding_dim
            self.ffn_embedding_dim = ffn_embedding_dim if ffn_embedding_dim else embedding_dim * 4
            self.num_attention_heads = num_attention_heads if num_attention_heads else embedding_dim // 64
        else:
            self.n_layers, self.embedding_dim, self.seq_len, self.num_attention_heads = MODEL_CONFIGS[model_name]
            self.ffn_embedding_dim = self.embedding_dim * 4
        self.n_devices = torch.cuda.device_count() if n_devices is None else n_devices
        self.placement_orders = placement_orders or list(range(self.n_devices))

    @classmethod
    def from_predefined_model(cls, model_name, n_devices=None, batch_size=1):
        n_layers, hidden_size, sequence_length, num_attention_heads = MODEL_CONFIGS[model_name]
        return cls(
            batch_size=batch_size,
            seq_len=sequence_length,
            n_layers=n_layers,
            embedding_dim=hidden_size,
            num_attention_heads=num_attention_heads,
            ffn_embedding_dim=hidden_size*4,
            n_devices=n_devices)

    @property
    def attention_heads_dim(self):
        return self.embedding_dim // self.num_attention_heads

    def create_layers_and_inputs(self):
        transformer_layers = [
            TransformerLayer(
                self.embedding_dim,
                self.ffn_embedding_dim,
                self.num_attention_heads
            )
            for _ in range(self.n_layers)
        ]
        x = torch.randn(self.seq_len, self.batch_size, self.embedding_dim)
        return transformer_layers, x

    def create_layers_gpu(self, device='cuda'):
        transformer_layers = [
            TransformerLayer(
                self.embedding_dim,
                self.ffn_embedding_dim,
                self.num_attention_heads,
                device=device
            )
            for _ in range(self.n_layers // self.n_devices)
        ]
        return transformer_layers

    def create_inputs(self, device='cuda', requires_grad=False):
        x = torch.randn(self.seq_len, self.batch_size, self.embedding_dim, device=device)
        return x.requires_grad_(requires_grad)

    def create_inputs_empty(self, device='cuda'):
        x = torch.empty((self.seq_len, self.batch_size, self.embedding_dim), device=device)
        return x

    def create_pseudo_inputs_outputs(self, device='cuda', inputs_requires_grad=False):
        x = torch.randn(self.seq_len, self.batch_size, self.embedding_dim, device=device)
        y = torch.cat([x[1:], torch.randn(1, self.batch_size, self.embedding_dim, device=device)])
        return x.requires_grad_(inputs_requires_grad), y

    def create_pseudo_attention_cache(self, length, device='cuda', requires_grad=False):
        size = (self.batch_size * self.num_attention_heads, length, self.attention_heads_dim)
        return [{
            'k': torch.randn(size, device=device).requires_grad_(requires_grad),
            'v': torch.randn(size, device=device).requires_grad_(requires_grad),
        } for _ in range(self.n_layers)]

    def compute_pseudo_loss(self, pred_y, y):
        return ((pred_y - y)**2).mean()

    def create_layers_and_inputs_on_gpu(self):
        assert self.n_layers % self.n_devices == 0
        print("Use placement orders: ", self.placement_orders)
        layers_per_device = self.n_layers // self.n_devices
        transformer_layers = [
            TransformerLayer(
                self.embedding_dim,
                self.ffn_embedding_dim,
                self.num_attention_heads,
                device='cuda:' + str(self.placement_orders[i // layers_per_device]),
            )
            for i in range(self.n_layers)
        ]
        x = torch.randn(self.seq_len, self.batch_size, self.embedding_dim, device='cuda:' + str(self.placement_orders[0]))
        return transformer_layers, x


class MultiheadLMAttentionWithCache(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True, device='cpu'):
        super().__init__()

        self.embed_dim = embed_dim

        self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias).to(device)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias).to(device)

        # TODO: initialize the weights correctly

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

    def forward(self, x, cache=None, batch_id=0, input_id=0, layer_id=0, slice_len=128):
        tgt_len, bsz, embed_dim = x.size()
        assert embed_dim == self.embed_dim
        q, k, v = self.in_proj(x).view(tgt_len, bsz * self.num_heads, self.head_dim * 3).transpose_(0, 1).chunk(3, dim=-1)
        # pytorch 1.7+ doesn't allow this inplace op
        q = q * self.scaling

        k_cache = cache.k[batch_id, layer_id, :, input_id * slice_len : (input_id + 1) * slice_len]
        v_cache = cache.v[batch_id, layer_id, :, input_id * slice_len : (input_id + 1) * slice_len]
        k_cache.copy_(k)
        v_cache.copy_(v)

        k = cache.k[batch_id, layer_id, :, : (input_id + 1) * slice_len]
        v = cache.v[batch_id, layer_id, :, : (input_id + 1) * slice_len]

        attn_mask = x.new_full((tgt_len, tgt_len), NEG_INF).triu_(1)
        if input_id > 0:
            attn_mask = torch.cat([x.new_zeros(tgt_len, input_id * slice_len), attn_mask], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        del k
        del q
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, (input_id + 1) * slice_len)
        attn_weights += attn_mask[None, :, :]
        del attn_mask
        # TODO: patch softmax to use https://stackoverflow.com/questions/53732209/torch-in-place-operations-to-save-memory-softmax
        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
        attn = torch.bmm(attn_probs, v)
        del v
        attn = attn.transpose_(0, 1).contiguous().view(tgt_len, bsz, -1)
        attn = self.out_proj(attn)
        return attn


class TransformerLayer(nn.Module):
    def __init__(self, embedding_dim, ffn_embedding_dim, num_attention_heads, device='cpu'):
        super().__init__()
        self.attn_ln = nn.LayerNorm(embedding_dim).to(device)
        self.attn = MultiheadLMAttentionWithCache(embedding_dim, num_attention_heads, device=device)
        self.fc_ln = nn.LayerNorm(embedding_dim).to(device)
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim).to(device)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim).to(device)

    def forward(self, x, attn_cache=None, batch_id=0, input_id=0, layer_id=0, slice_len=128):
        # ==============================================
        # Per device per layer memory usage (GPT-3 137B)
        # ==============================================
        #
        # ~~~~~~~~~~~
        # Activations
        # ~~~~~~~~~~~
        #
        # C := batch_size * seqlen * hidden_size * sizeof(float16)
        # M := Megatron-LM model parallel size
        # 
        # Total activation: (4 + 73 / M) C
        # If M = 8, then the total activation is 630 MB per batch (678 MB if considering the input).
        # attn_weights & attn_probs takes 61% of the total activation size.
        # LayerNorm takes 15%
        #
        # Self-attention: (2 + 69 / M) C
        #   LayerNorm: C
        #   QKV_in_proj: 3C / M
        #   Q_scale: C / M
        #   attn_weights: 16C / M
        #   attn_probs: 48C / M
        #   attn: C / M
        #   output: C
        #
        # Feed-forward: (2 + 4 / M) C
        #   LayerNorm: C
        #   FC1: 4C / M
        #   FC2: C
        #

        y = x
        x = self.attn_ln(x)
        x = self.attn(x, attn_cache, batch_id=batch_id, input_id=input_id, layer_id=layer_id, slice_len=slice_len)
        x += y

        y = x
        x = self.fc_ln(x)
        x = self.fc1(x).relu_()
        x = self.fc2(x)
        x += y
        return x


class ModelParallelMultiheadLMAttentionWithCache(MultiheadLMAttentionWithCache):
    def __init__(self, embed_dim, num_heads, bias=True, device='cpu'):
        nn.Module.__init__(self)

        self.embed_dim = embed_dim

        self.in_proj = mpu.ColumnParallelLinear(embed_dim, 3 * embed_dim, bias=bias,
                                                gather_output=False, initialize_locally=True, device=device)
        self.out_proj = mpu.RowParallelLinear(embed_dim, embed_dim, bias=bias,
                                              input_is_parallel=True, initialize_locally=True, device=device)

        self.model_parallel_size = mpu.get_model_parallel_world_size()

        self.num_total_heads = num_heads
        self.num_heads = self.num_total_heads // self.model_parallel_size
        assert (
                self.num_heads * self.model_parallel_size == num_heads
        ), "Number of heads must be divisble by model parallel size"

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5


class ModelParallelTransformerLayer(TransformerLayer):
    def __init__(self, embedding_dim, ffn_embedding_dim, num_attention_heads, device='cpu'):
        nn.Module.__init__(self)
        self.model_parallel_size = mpu.get_model_parallel_world_size()
        assert ffn_embedding_dim % self.model_parallel_size == 0

        # TODO: write a custom inplace LayerNorm layer
        self.attn_ln = nn.LayerNorm(embedding_dim).to(device)
        self.attn = ModelParallelMultiheadLMAttentionWithCache(embedding_dim, num_attention_heads, device=device)
        self.fc_ln = nn.LayerNorm(embedding_dim).to(device)
        self.fc1 = mpu.ColumnParallelLinear(embedding_dim, ffn_embedding_dim, gather_output=False).to(device)
        self.fc2 = mpu.RowParallelLinear(ffn_embedding_dim, embedding_dim, input_is_parallel=True).to(device)


class SingleDeviceTransformer(nn.Module):
    def __init__(self, transformer_layers):
        super().__init__()
        self.layers = nn.ModuleList(transformer_layers)

    def forward(self, x, attn_caches=None):
        if attn_caches is None:
            attn_caches = [None] * len(self.layers)
        new_attn_caches = []
        for layer, attn_cache in zip(self.layers, attn_caches):
            x, new_attn_cache = layer(x, attn_cache)
            new_attn_caches.append(new_attn_cache)
        return x, new_attn_caches


class PipelinedTransformer(nn.Module):
    def __init__(self, nested_transformer_layers, config):
        super().__init__()
        self.config = config
        assert len(nested_transformer_layers) == config.n_devices
        self.single_device_transformers = nn.ModuleList([
            SingleDeviceTransformer(transformer_layers) for transformer_layers in nested_transformer_layers
        ])

    def forward(self, segmented_xs):
        n_segments = len(segmented_xs)

        def _worker(device_id, model, my_queue, succ_queue):
            with torch.cuda.device(device_id):
                cache = None
                for t in range(n_segments):
                    x = my_queue.get().to(device_id, non_blocking=True)
                    x, cache = model(x, cache)
                    succ_queue.put(x)

        all_queues = [queue.Queue() for _ in range(self.config.n_devices + 1)]
        threads = [threading.Thread(target=_worker,
                                    args=(self.config.placement_orders[i], self.single_device_transformers[i],
                                          all_queues[i], all_queues[i + 1]))
                   for i in range(self.config.n_devices)]
        for x in segmented_xs:
            all_queues[0].put(x)
        for thread in threads:
            thread.start()
        results = []
        for _ in range(n_segments):
            results.append(all_queues[-1].get())
        for thread in threads:
            thread.join()
        return results


def save_layers_and_inputs(layers, grad_layers, layer_ids, inputs, prefix):
    for i, layer, grad in zip(layer_ids, layers, grad_layers):
        torch.save(layer.state_dict(), ".".join([prefix, str(i)]))
        torch.save(grad, ".".join([prefix, 'grad', str(i)]))
    torch.save(inputs, ".".join([prefix, "inputs"]))


def load_layers(layers, layer_ids, prefix):
    for i, layer in zip(layer_ids, layers):
        layer.load_state_dict(torch.load(".".join([prefix, str(i)])))


def load_grads(layer_ids, prefix):
    all_grads = []
    for i in layer_ids:
        all_grads.append(torch.load(".".join([prefix, 'grad', str(i)])))
    return all_grads


def load_inputs(prefix):
    return torch.load(".".join([prefix, "inputs"]))


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


def uniform_slice_batch_and_input(x, n_batch_slices, n_input_slices):
    seq_len, batch_size, _ = x.size()
    sliced_batch = np.empty((n_batch_slices, n_input_slices), dtype='O')
    start_batch_index = 0
    for i in range(n_batch_slices):
        batch_size_slice = batch_size // n_batch_slices + int(i < batch_size % n_batch_slices)
        start_input_index = 0
        for j in range(n_input_slices):
            seq_len_slice = seq_len // n_input_slices + int(j < seq_len % n_input_slices)
            sliced_batch[i, j] = (x[start_input_index:start_input_index + seq_len_slice,
                                  start_batch_index:start_batch_index + batch_size_slice].contiguous())
            start_input_index += seq_len_slice
        assert start_input_index == seq_len
        start_batch_index += batch_size_slice
    assert start_batch_index == batch_size
    return sliced_batch


def uniform_slice_layers(transformer_layers, n_devices=None):
    n_layers = len(transformer_layers)
    n_devices = n_devices if n_devices else torch.cuda.device_count()
    nested_layers = []
    layer_idx = 0
    for i in range(n_devices):
        n_layers_device = n_layers // n_devices + int(i < n_layers % n_devices)
        nested_layers.append(transformer_layers[layer_idx:layer_idx + n_layers_device])
        layer_idx += n_layers_device
    assert layer_idx == n_layers
    return nested_layers
