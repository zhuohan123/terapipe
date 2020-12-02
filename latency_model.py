import json

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from transformer_models import MODEL_CONFIGS

SCAN_GRID = (16, 16)
STEP_GAP = 8


class SingleLayerLatency:
    def __init__(self, f_plus_b_array, update_array, attn_cache_linear_model):
        self.f_plus_b_array = f_plus_b_array
        self.update_array = update_array
        self.attn_cache_linear_model = attn_cache_linear_model

    def predict(self, seqlen, attn_cache_len):
        assert seqlen % 8 == 0
        f_and_b_time = self.f_plus_b_array[seqlen // 8]
        update_time = self.update_array[seqlen // 8]
        attn_time = self.attn_cache_linear_model.predict([[seqlen, attn_cache_len, seqlen * attn_cache_len]])
        return f_and_b_time + update_time + attn_time


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


if __name__ == "__main__":
    # analysis_model('gpt3-175b')
    single_layer_model = fit_single_layer_model('gpt3-175b')
    print(single_layer_model.predict(128, 32))
