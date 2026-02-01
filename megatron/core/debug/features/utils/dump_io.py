# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Shared IO helpers for tensor/grad dumps."""

from typing import Optional, Tuple
import os


def get_rank_info() -> Tuple[int, int, int, int]:
    """Return (tp_rank, pp_rank, ep_rank, edp_rank)."""
    from megatron.core import parallel_state as mpu

    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    try:
        ep_rank = mpu.get_expert_model_parallel_rank()
    except (AssertionError, RuntimeError):
        ep_rank = 0
    try:
        edp_rank = mpu.get_expert_data_parallel_rank()
    except (AssertionError, RuntimeError):
        edp_rank = mpu.get_data_parallel_rank()
    return tp_rank, pp_rank, ep_rank, edp_rank


def should_save_edp(rank_info: Optional[Tuple[int, int, int, int]] = None) -> bool:
    if rank_info is None:
        rank_info = get_rank_info()
    return rank_info[3] == 0


def get_tensor_dump_iter_dir(
    save_dir: str,
    iteration: int,
    rank_info: Optional[Tuple[int, int, int, int]] = None,
) -> str:
    if rank_info is None:
        rank_info = get_rank_info()
    tp_rank, pp_rank, ep_rank, _ = rank_info
    iter_dir = os.path.join(
        save_dir,
        f"iter_{iteration:07d}",
        f"mp_rank_{tp_rank:02d}_{pp_rank:03d}_{ep_rank:03d}",
    )
    os.makedirs(iter_dir, exist_ok=True)
    return iter_dir


def get_wgrad_iter_dir(save_dir: str, iteration: int) -> str:
    iter_dir = os.path.join(save_dir, "wgrads", f"iter_{iteration:07d}")
    os.makedirs(iter_dir, exist_ok=True)
    return iter_dir


def get_wgrad_checkpoint_name(
    rank_info: Optional[Tuple[int, int, int, int]] = None,
) -> str:
    if rank_info is None:
        rank_info = get_rank_info()
    tp_rank, pp_rank, ep_rank, _ = rank_info

    name = f"mp_rank_{tp_rank:02d}"
    from megatron.core import parallel_state as mpu
    try:
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            name += f"_{pp_rank:03d}"
    except (AssertionError, RuntimeError):
        pass
    try:
        if mpu.get_expert_model_parallel_world_size() > 1:
            name += f"_{ep_rank:03d}"
    except (AssertionError, RuntimeError):
        pass
    return name
