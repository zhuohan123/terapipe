import json
import time
import numba

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from transformer_models import MODEL_CONFIGS

SCAN_GRID = (16, 16, 4)
STEP_GAP = 8


class SingleLayerLatency:
    def __init__(self, f_plus_b_array, update_array, attn_cache_linear_model):
        self.f_plus_b_array = f_plus_b_array
        self._update_time = np.mean(update_array)
        self.attn_cache_linear_model = attn_cache_linear_model

    def predict(self, seqlen, attn_cache_len):
        assert seqlen % STEP_GAP == 0
        f_and_b_time = self.f_plus_b_array[seqlen // STEP_GAP - 1]
        
        attn_time = self.attn_cache_linear_model.predict([[seqlen, attn_cache_len, seqlen * attn_cache_len]])[0]
        return f_and_b_time + attn_time

    def update_time(self):
        return self._update_time

    def predict_latency_grid(self, total_batch_size, total_length):
        assert total_length % STEP_GAP == 0
        n_seq_slices = total_length // STEP_GAP
        grid = np.zeros((1 + total_batch_size, 1 + n_seq_slices, 1 + n_seq_slices), dtype=np.float)
        for batch_size in range(1, total_batch_size + 1):
            for seqlen in range(1, n_seq_slices + 1):
                for cachelen in range(n_seq_slices + 1):
                    grid[batch_size, seqlen, cachelen] = batch_size * self.predict(seqlen * STEP_GAP, cachelen * STEP_GAP)
        return grid


def merge_dict(data):
    keys = list(data[0].keys())
    result = {k: [] for k in keys}
    for x in data:
        for k in keys:
            result[k].append(x[k])
    for k in keys:
        result[k] = np.array(result[k])
    return result


def fit_single_layer_model(model_name):
    n_layers, hidden_size, seqlen, num_attention_heads = MODEL_CONFIGS[model_name]
    with open(f"{model_name}.latency_model.attn_cache_len.json") as f:
        attn_cache_len_latency = json.load(f)
    with open(f"{model_name}.latency_model.seqlen.json") as f:
        seqlen_latency = json.load(f)
    attn_cache_len_latency = merge_dict(attn_cache_len_latency)
    seqlen_latency = merge_dict(seqlen_latency)

    X = np.arange(STEP_GAP, seqlen+1, STEP_GAP)
    attn_cache_len_X = np.arange(seqlen // SCAN_GRID[1], seqlen + 1, seqlen // SCAN_GRID[1])
    seqlen_X = np.arange(seqlen // SCAN_GRID[0], seqlen + 1, seqlen // SCAN_GRID[0])

    f_plus_b = seqlen_latency['forward_mean'] + seqlen_latency['backward_mean']

    grid_seqlen_skip_gap = seqlen // SCAN_GRID[0]

    f_plus_b_for_attn_cache_len = f_plus_b[X % grid_seqlen_skip_gap == 0]

    attn_f_plus_b = attn_cache_len_latency['forward_mean'] + attn_cache_len_latency['backward_mean']
    attn_f_plus_b = attn_f_plus_b.reshape(SCAN_GRID)
    attn_delta = attn_f_plus_b - f_plus_b_for_attn_cache_len[None, :]

    # create dataset
    X_seqlen = np.tile(seqlen_X, (SCAN_GRID[1], 1)).ravel()
    X_attn_cache_len = np.tile(attn_cache_len_X, (1, SCAN_GRID[0])).ravel()
    X_multiplied = (seqlen_X[None, :] * attn_cache_len_X[:, None]).ravel()
    X_data = np.stack([X_seqlen, X_attn_cache_len, X_multiplied]).transpose()
    Y_data = attn_delta.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.33, random_state=42)

    attention_cache_len_lrmodel = LinearRegression()
    attention_cache_len_lrmodel.fit(X_train, y_train)
    # print(attention_cache_len_lrmodel.coef_, attention_cache_len_lrmodel.intercept_)
    print("Linear regression score:", attention_cache_len_lrmodel.score(X_test, y_test))
    return SingleLayerLatency(f_plus_b, seqlen_latency['update_mean'], attention_cache_len_lrmodel)


@numba.jit(nopython=True, parallel=True)
def planning(latency_grid, total_batch_size, total_seqlen, pipelinelen, max_latency):
    assert total_seqlen % STEP_GAP == 0
    total_nslices = total_seqlen // STEP_GAP
    # f[bs][acc_nslices]: Optimal latency with the given batch size and accumulated nslices
    f = np.zeros((total_batch_size + 1, total_nslices + 1), dtype=np.float64)
    f_step = np.zeros((total_batch_size + 1, total_nslices + 1), dtype=np.int64)
    f_max_latency = np.zeros((total_batch_size + 1, total_nslices + 1), dtype=np.float64)
    # DP[TOTAL_LENGTH]
    for bs in range(1, total_batch_size + 1):
        for acc_nslices in range(1, total_nslices + 1):
            f[bs, acc_nslices] = np.inf
            for step_nslices in range(1, acc_nslices + 1):
                step_latency = latency_grid[bs, step_nslices, acc_nslices - step_nslices]
                total_time = f[bs, acc_nslices - step_nslices] + step_latency
                if step_latency <= max_latency and total_time < f[bs, acc_nslices]:
                    f[bs, acc_nslices] = total_time
                    f_step[bs, acc_nslices] = step_nslices
                    f_max_latency[bs, acc_nslices] = max(f_max_latency[bs, acc_nslices], step_latency)

    f_last = f[:, total_nslices]
    f_last_max_latency = f_max_latency[:, total_nslices]

    # g[acc_bs]: Optimal latency with the given accumulated batch size
    g = np.zeros(total_batch_size + 1, dtype=np.float64)
    g_step = np.zeros(total_batch_size + 1, dtype=np.int64)
    g_max_latency = np.zeros(total_batch_size + 1, dtype=np.float64)

    for acc_bs in range(1, total_batch_size + 1):
        g[acc_bs] = np.inf
        for bs in range(1, acc_bs + 1):
            total_time = g[acc_bs - bs] + f_last[bs]
            if total_time < g[acc_bs]:
                g[acc_bs] = total_time
                g_step[acc_bs] = bs
                g_max_latency[acc_bs] = max(g_max_latency[acc_bs], f_last_max_latency[bs])

    final_time = (pipelinelen - 1) * g_max_latency[total_batch_size] + g[total_batch_size]
    if final_time == np.inf:
        return final_time, None
    all_slice_scheme = []
    current_batch_size = total_batch_size
    while current_batch_size > 0:
        this_step_batch_size = g_step[current_batch_size]
        slice_scheme = []
        current_len = total_nslices
        while current_len > 0:
            this_step_len = f_step[this_step_batch_size, current_len]
            slice_scheme.append(this_step_len * STEP_GAP)
            current_len -= this_step_len
        all_slice_scheme.append((this_step_batch_size, slice_scheme))
        current_batch_size -= this_step_batch_size
    return final_time, all_slice_scheme


def evaluate_split(latency_model, split_scheme, pipelinelen):
    cache_len = 0
    total_time = 0
    largest_time = 0
    for split in split_scheme:
        split_time = latency_model.predict(split, cache_len)
        total_time += split_time
        cache_len += split
        largest_time = max(largest_time, split_time)
    return total_time + (pipelinelen - 1) * largest_time


if __name__ == "__main__":
    # analysis_model('gpt3-175b')
    single_layer_model = fit_single_layer_model('gpt3-175b')
    total_batch_size = 2
    seqlen = 2048
    pipelinelen = 48
    time_grid = single_layer_model.predict_latency_grid(total_batch_size, seqlen)
    all_possible_latencies = np.sort(np.unique(time_grid))
    best_latency = np.inf
    best_scheme = None
    for max_latency in all_possible_latencies:
        if max_latency * pipelinelen > best_latency:
            break
        latency, all_split_scheme = planning(time_grid, total_batch_size, seqlen, pipelinelen, max_latency)
        if all_split_scheme is not None:
            all_split_scheme = [(batch_size, list(reversed(split_scheme))) for batch_size, split_scheme in all_split_scheme]
        if latency < best_latency:
            best_latency = latency
            best_scheme = all_split_scheme
            print(best_latency, best_scheme, len(best_scheme))
