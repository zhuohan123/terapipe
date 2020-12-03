import json
import time
import numba

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from transformer_models import MODEL_CONFIGS

SCAN_GRID = (16, 16)
STEP_GAP = 8


class SingleLayerLatency:
    def __init__(self, f_plus_b_array, update_array, attn_cache_linear_model):
        self.f_plus_b_array = f_plus_b_array
        self.update_time = np.mean(update_array)
        self.attn_cache_linear_model = attn_cache_linear_model

    def predict(self, seqlen, attn_cache_len):
        assert seqlen % STEP_GAP == 0
        f_and_b_time = self.f_plus_b_array[seqlen // STEP_GAP - 1]
        
        attn_time = self.attn_cache_linear_model.predict([[seqlen, attn_cache_len, seqlen * attn_cache_len]])[0]
        return f_and_b_time + attn_time

    def update_time(self):
        return self.update_time

    def predict_latency_grid(self, total_length):
        assert total_length % STEP_GAP == 0
        n_seq_slices = total_length // STEP_GAP
        grid = np.zeros((1 + n_seq_slices, 1 + n_seq_slices), dtype=np.float)
        for seqlen in range(1, n_seq_slices + 1):
            for cachelen in range(n_seq_slices + 1):
                grid[seqlen, cachelen] = self.predict(seqlen * STEP_GAP, cachelen * STEP_GAP)
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
def planning(latency_grid, seqlen, pipelinelen, max_latency):
    assert seqlen % STEP_GAP == 0
    n_seq_slices = seqlen // STEP_GAP
    dp = np.zeros(n_seq_slices + 1, dtype=np.float64)
    dp_step_len = np.zeros(n_seq_slices + 1, dtype=np.int64)
    dp_actual_max_latency = np.zeros(n_seq_slices + 1, dtype=np.float64)
    # DP[TOTAL_LENGTH]
    for total_length in range(1, n_seq_slices + 1):
        dp[total_length] = np.inf
        for this_step_length in range(1, total_length + 1):
            step_latency = latency_grid[this_step_length, total_length - this_step_length]
            total_time = dp[total_length - this_step_length] + step_latency
            if step_latency <= max_latency and total_time < dp[total_length]:
                dp[total_length] = total_time
                dp_step_len[total_length] = this_step_length
                dp_actual_max_latency[total_length] = max(dp_actual_max_latency[total_length], step_latency)

    final_time = (pipelinelen - 1) * dp_actual_max_latency[n_seq_slices] + dp[n_seq_slices]
    if final_time == np.inf:
        return final_time, None
    slice_scheme = []
    current_len = n_seq_slices
    while current_len > 0:
        this_step_len = dp_step_len[current_len]
        slice_scheme.append(this_step_len * STEP_GAP)
        current_len -= this_step_len
    return final_time, slice_scheme


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
    seqlen = 2048
    pipelinelen = 48
    time_grid = single_layer_model.predict_latency_grid(seqlen)
    all_possible_latencies = np.sort(np.unique(time_grid))
    best_latency = np.inf
    best_scheme = None
    for max_latency in all_possible_latencies:
        if max_latency * pipelinelen > best_latency:
            break
        latency, split_scheme = planning(time_grid, seqlen, pipelinelen, max_latency)
        if split_scheme is not None:
            split_scheme = list(reversed(split_scheme))
        if latency < best_latency:
            best_latency = latency
            best_scheme = split_scheme
            print(best_latency, best_scheme, len(best_scheme))
