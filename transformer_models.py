import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import queue
import mpu

NEG_INF = -1e10


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
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim if ffn_embedding_dim else embedding_dim * 4
        self.num_attention_heads = num_attention_heads if num_attention_heads else embedding_dim // 64
        self.n_devices = torch.cuda.device_count() if n_devices is None else n_devices
        assert self.n_devices <= torch.cuda.device_count()
        self.placement_orders = placement_orders or list(range(self.n_devices))

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

    def forward(self, x, cache=None):
        tgt_len, bsz, embed_dim = x.size()
        assert embed_dim == self.embed_dim
        q, k, v = self.in_proj(x).contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim * 3).transpose(0, 1).chunk(3, dim=-1)
        q *= self.scaling
        attn_mask = x.new_full((tgt_len, tgt_len), NEG_INF).triu(1)
        src_len = tgt_len
        if cache is not None:
            cache_len = cache["k"].size()[1]
            k = torch.cat([cache["k"], k], dim=1)
            v = torch.cat([cache["v"], v], dim=1)
            attn_mask = torch.cat([x.new_zeros(tgt_len, cache_len), attn_mask], dim=1)
            src_len += cache_len

        new_cache = {
            "k": k,
            "v": v,
        }

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        attn_weights += attn_mask[None, :, :]
        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        attn = self.out_proj(attn)

        return attn, new_cache


class TransformerLayer(nn.Module):
    def __init__(self, embedding_dim, ffn_embedding_dim, num_attention_heads, device='cpu'):
        super().__init__()
        self.attn_ln = nn.LayerNorm(embedding_dim).to(device)
        self.attn = MultiheadLMAttentionWithCache(embedding_dim, num_attention_heads, device=device)
        self.fc_ln = nn.LayerNorm(embedding_dim).to(device)
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim).to(device)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim).to(device)

    def forward(self, x, attn_cache=None):
        seq_len, batch_size, _ = x.size()
        y = x
        x = self.attn_ln(x)
        x, new_attn_cache = self.attn(x, attn_cache)
        x = y + x
        y = x
        x = self.fc_ln(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = y + x
        return x, new_attn_cache


class ModelParallelMultiheadLMAttentionWithCache(MultiheadLMAttentionWithCache):
    def __init__(self, embed_dim, num_heads, bias=True, device='cpu'):
        nn.Module.__init__(self)

        self.embed_dim = embed_dim

        self.in_proj = mpu.ColumnParallelLinear(embed_dim, 3 * embed_dim, bias=bias, gather_output=False).to(device)
        self.out_proj = mpu.RowParallelLinear(embed_dim, embed_dim, bias=bias, input_is_parallel=True).to(device)

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


class StreamCopyQueue:
    def __init__(self, next_device):
        self.stream = torch.cuda.Stream()
        self.queue = queue.Queue()
        self.next_device = next_device

    def put(self, x):
        with torch.cuda.stream(self.stream):
            x = x.to(device_id, non_blocking=True)
            event = self.stream.record_event()
        self.queue.put((x, event))

    def get(self):
        x, event = self.queue.get()
        torch.cuda.current_stream().wait_event(event)
        return x


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
                    x = my_queue.get()
                    x, cache = model(x, cache)
                    succ_queue.put(x)
        all_queues = [queue.Queue()]
        all_queues += [StreamCopyQueue(torch.device(f'cuda:{self.config.placement_orders[i]}')) for i in range(1, self.config.n_devices)]
        all_queues.append(queue.Queue())
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
