import json
import time
import numba

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from transformer_models import MODEL_CONFIGS, BATCH_CONFIGS

SCAN_GRID = (16, 16, 4)
STEP_GAP = 8


class SingleLayerLatency:
    def __init__(self, no_content_performance, update_time, context_len_lrmodel, comm_time):
        self.no_content_performance = no_content_performance
        self.update_time = update_time
        self.context_len_lrmodel = context_len_lrmodel
        self.comm_time = comm_time

        # generate latency grid
        X = self._generate_model_input()
        batch, seqlen = self.no_content_performance.shape
        y = self.context_len_lrmodel.predict(X).reshape(batch, seqlen, seqlen)
        b = self.no_content_performance + self.comm_time * 2
        self.latency_grid = b.reshape(batch, seqlen, 1) + y

    def predict(self, batch_size, seqlen, context_len):
        assert seqlen % STEP_GAP == 0
        no_context_time = self.no_content_performance[seqlen // STEP_GAP - 1]
        context_time = self.context_len_lrmodel.predict(
            [[batch_size * seqlen, batch_size * context_len, batch_size * seqlen * context_len]])[0]
        return no_context_time + context_time

    def _generate_model_input(self):
        batch, seqlen = self.no_content_performance.shape
        x, y, z = np.where(np.ones((batch, seqlen, seqlen)))
        y *= STEP_GAP
        z *= STEP_GAP
        u = x * y * z
        y *= x
        z *= x
        return np.stack([y, z, u]).transpose()


def parse_json(r):
    results = {}
    for k, v in r.items():
        key = tuple(map(int, k.split('_')))
        results[key] = v
    return results


def fit_single_layer_model(model_name, model_parallel_size):
    with open(f'performance_model_data/latency_model.{model_name}.mp_{model_parallel_size}.json', 'r') as f:
        data = parse_json(json.load(f))
    # data: (batch_size, seqlen, attention_cache_len) -> time
    n_layers, hidden_size, seqlen, num_attention_heads = MODEL_CONFIGS[model_name]
    batch_size = BATCH_CONFIGS[model_name]
    # optimizer step time for the model
    update_time = np.mean([t['update_mean'] for t in data.values()])

    no_context_len_data = {k[:2]:v for k, v in data.items() if k[2] == 0}
    no_content_performance = np.full((batch_size + 1, seqlen // STEP_GAP + 1), np.inf)
    for (b, s), t in no_context_len_data.items():
        no_content_performance[b, s // STEP_GAP] = t['forward_mean'] + t['backward_mean']

    context_len_data = {k:v for k, v in data.items() if k[2] > 0}
    X = []
    y = []
    for (b, s, c), t in context_len_data.items():
        X.append([b*s, b*c, b*s*c])
        y.append(t['forward_mean'] + t['backward_mean'] - no_content_performance[b, s // STEP_GAP])

    X = np.array(X)
    y = np.array(y)
    # create dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    context_len_lrmodel = LinearRegression()
    context_len_lrmodel.fit(X_train, y_train)
    # print(context_len_lrmodel.coef_, context_len_lrmodel.intercept_)
    print("Linear regression score:", context_len_lrmodel.score(X_test, y_test))

    # communication performance
    with open(f'performance_model_data/{model_name}.communication_latency.json', 'r') as f:
        comm_data = json.load(f)
        xp = np.array(comm_data['tensor_sizes'])
        yp = np.array(comm_data['mean'])
    x = np.arange(STEP_GAP, batch_size * seqlen + 1, STEP_GAP)
    c = np.interp(x, xp, yp).reshape(batch_size, seqlen // STEP_GAP)
    comm_time = np.zeros((batch_size + 1, seqlen // STEP_GAP + 1))
    comm_time[1:, 1:] = c  # pad with zero
    return SingleLayerLatency(no_content_performance, update_time, context_len_lrmodel, comm_time)


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
    single_layer_model = fit_single_layer_model('gpt3-175b', 8)
    total_batch_size = 2
    seqlen = 2048
    pipelinelen = 48
    time_grid = single_layer_model.latency_grid
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
