# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Model and data parallel groups."""

import torch

from .utils import ensure_divisibility

_INITIALIZED = False

_RANK = None
_WORLD_SIZE = None
_DATA_PARALLEL_SIZE = None
_MODEL_PARALLEL_SIZE = None
_PIPELINE_PARALLEL_SIZE = None

_MODEL_PARALLEL_GROUP = None
_MODEL_PARALLEL_GROUP_RANK = -1

_PIPELINE_PARALLEL_GROUP_RANK = None

_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP_RANK = None


def initialize_model_parallel(model_parallel_size, pipeline_parallel_size=1, embedding_parallel_size=0):
    """
    Initialize model parallel and pipeline parallel groups.
    :param model_parallel_size: Size of the model parallel group.
    :param pipeline_parallel_size: Pipeline length.
    :param embedding_parallel_size: Size of the embedding parallel group.
    For example, if we have 17 GPUs in total, and have model_parallel_size = 4,
    pipeline_parallel_size = 2, embedding_parallel_size=1,
    all GPUs will be grouped in the following way:
                      Model:
    Pipeline:   [0] -> [[  1,  2,  3,  4] -> [  5,  6,  7,  8]] -> [0]
    Data:                  |   |   |   |        |   |   |   |
                [0] -> [[  9,  10, 11, 12] -> [ 13, 14, 15, 16]] -> [0]
    """
    global _INITIALIZED
    assert not _INITIALIZED
    _INITIALIZED = True

    if torch.distributed.get_rank() == 0:
        print('> initializing with model parallel size {} and pipeline parallel size {}'.format(
            model_parallel_size, pipeline_parallel_size))
    global _RANK, _WORLD_SIZE, _DATA_PARALLEL_SIZE, _MODEL_PARALLEL_SIZE, _PIPELINE_PARALLEL_SIZE, _EMBEDDING_PARALLEL_SIZE, _OFFSET_RANK
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    total_model_parallel_size = model_parallel_size * pipeline_parallel_size
    ensure_divisibility(world_size - embedding_parallel_size, total_model_parallel_size)
    rank = torch.distributed.get_rank()

    _RANK = rank
    _WORLD_SIZE = world_size
    _DATA_PARALLEL_SIZE = world_size // total_model_parallel_size
    _MODEL_PARALLEL_SIZE = model_parallel_size
    _PIPELINE_PARALLEL_SIZE = pipeline_parallel_size
    _EMBEDDING_PARALLEL_SIZE = embedding_parallel_size

    offset_rank = rank - embedding_parallel_size

    _OFFSET_RANK = offset_rank

    # Build the data parallel groups.
    global _DATA_PARALLEL_GROUP, _DATA_PARALLEL_GROUP_RANK
    for i in range(embedding_parallel_size, total_model_parallel_size + embedding_parallel_size):
        ranks = range(i, world_size, total_model_parallel_size)
        group = torch.distributed.new_group(ranks)
        
        if i == (_OFFSET_RANK % total_model_parallel_size):
            _DATA_PARALLEL_GROUP = group
            _DATA_PARALLEL_GROUP_RANK = _OFFSET_RANK // total_model_parallel_size

    # Build the model parallel groups.
    global _MODEL_PARALLEL_GROUP, _MODEL_PARALLEL_GROUP_RANK
    for i in range(world_size // model_parallel_size):
        ranks = range(i * model_parallel_size + embedding_parallel_size,
                      (i + 1) * model_parallel_size + embedding_parallel_size)
        group = torch.distributed.new_group(ranks)
        if i == (_OFFSET_RANK // model_parallel_size):
            _MODEL_PARALLEL_GROUP = group
            _MODEL_PARALLEL_GROUP_RANK = i % model_parallel_size

    global _PIPELINE_PARALLEL_GROUP_RANK

    _PIPELINE_PARALLEL_GROUP_RANK = _OFFSET_RANK // model_parallel_size % pipeline_parallel_size

    # Build the embedding parallel group.
    if embedding_parallel_size > 0:
        global _EMBEDDING_PARALLEL_GROUP
        ranks = range(embedding_parallel_size)
        group = torch.distributed.new_group(ranks)
        if _RANK < embedding_parallel_size:
            _EMBEDDING_PARALLEL_GROUP = group


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    return _INITIALIZED


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _INITIALIZED
    return _MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _INITIALIZED
    return _DATA_PARALLEL_GROUP


def get_model_parallel_world_size():
    """Return world size for the model parallel group."""
    assert _INITIALIZED
    return _MODEL_PARALLEL_SIZE


def get_model_parallel_rank():
    assert _INITIALIZED
    return _MODEL_PARALLEL_GROUP_RANK


def get_model_parallel_src_rank():
    assert _INITIALIZED
    return (_OFFSET_RANK // _MODEL_PARALLEL_SIZE) * _MODEL_PARALLEL_SIZE + _EMBEDDING_PARALLEL_SIZE


def get_model_parallel_dst_rank():
    assert _INITIALIZED
    return (_OFFSET_RANK // _MODEL_PARALLEL_SIZE) * _MODEL_PARALLEL_SIZE + _MODEL_PARALLEL_SIZE - 1 + _EMBEDDING_PARALLEL_SIZE


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    assert _INITIALIZED
    return _DATA_PARALLEL_SIZE


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    assert _INITIALIZED
    return _DATA_PARALLEL_GROUP_RANK


def get_pipeline_parallel_group_rank():
    """Return my group rank pipeline parallel."""
    assert _INITIALIZED
    return _PIPELINE_PARALLEL_GROUP_RANK


def destroy_model_parallel():
    """Set the groups to none."""
    global _INITIALIZED
    _INITIALIZED = False
    global _RANK
    global _WORLD_SIZE
    global _DATA_PARALLEL_SIZE
    global _MODEL_PARALLEL_SIZE
    global _PIPELINE_PARALLEL_SIZE

    global _MODEL_PARALLEL_GROUP
    global _MODEL_PARALLEL_GROUP_ID
    global _MODEL_PARALLEL_GROUP_RANK

    global _PIPELINE_PARALLEL_GROUP_ID
    global _PIPELINE_PARALLEL_GROUP_RANK

    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_ID
    global _DATA_PARALLEL_GROUP_RANK

    global _EMBEDDING_PARALLEL_GROUP

    _RANK = None
    _WORLD_SIZE = None
    _DATA_PARALLEL_SIZE = None
    _MODEL_PARALLEL_SIZE = None
    _PIPELINE_PARALLEL_SIZE = None

    _MODEL_PARALLEL_GROUP = None
    _MODEL_PARALLEL_GROUP_ID = None
    _MODEL_PARALLEL_GROUP_RANK = -1

    _PIPELINE_PARALLEL_GROUP_ID = None
    _PIPELINE_PARALLEL_GROUP_RANK = None

    _DATA_PARALLEL_GROUP = None
    _DATA_PARALLEL_GROUP_ID = None
    _DATA_PARALLEL_GROUP_RANK = None

    _EMBEDDING_PARALLEL_GROUP = None
