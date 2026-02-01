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

"""Tensor dump helpers."""

import logging
import os
from typing import Optional, Tuple

import torch

from megatron.core.debug.features.utils.dump_io import (
    get_rank_info as _get_rank_info_impl,
    get_tensor_dump_iter_dir,
    should_save_edp,
)

logger = logging.getLogger(__name__)


class TensorDumpState:
    """State for tensor dumps."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.current_iteration: Optional[int] = None
        self.save_dir: Optional[str] = None
        self._tensor_count: int = 0
        self._microbatch_indices: dict = {}

    def has_data(self) -> bool:
        return self._tensor_count > 0

    def increment_count(self):
        self._tensor_count += 1

    def get_microbatch_index(self, layer_name: str, tensor_name: str) -> int:
        key = (layer_name, tensor_name)
        idx = self._microbatch_indices.get(key, 0)
        self._microbatch_indices[key] = idx + 1
        return idx

    @property
    def tensor_count(self) -> int:
        return self._tensor_count


TENSOR_DUMP_STATE = TensorDumpState()


def _get_rank_info() -> Tuple[int, int, int, int]:
    return _get_rank_info_impl()


def _should_save() -> bool:
    """Return True for expert DP rank 0."""
    return should_save_edp()


def _get_iter_dir(save_dir: str, iteration: int) -> str:
    return get_tensor_dump_iter_dir(
        save_dir,
        iteration,
        rank_info=_get_rank_info(),
    )


def _sanitize_name(name: str) -> str:
    return name.replace("/", "__").replace("\\", "__").replace(".", "_")


def save_tensor_direct(
    save_dir: str,
    iteration: int,
    layer_name: str,
    tensor_name: str,
    tensor: torch.Tensor,
    log_to_metrics: bool = True,
    microbatch_idx: Optional[int] = None,
    include_microbatch: bool = True,
    track_state: bool = True,
) -> None:
    if not _should_save():
        return

    iter_dir = _get_iter_dir(save_dir, iteration)
    if include_microbatch:
        if microbatch_idx is None:
            microbatch_idx = (
                TENSOR_DUMP_STATE.get_microbatch_index(layer_name, tensor_name)
                if track_state
                else 0
            )

    safe_layer = _sanitize_name(layer_name)
    safe_tensor = _sanitize_name(tensor_name)
    if include_microbatch:
        filename = f"{safe_layer}__{safe_tensor}__mb{microbatch_idx:02d}.pt"
    else:
        filename = f"{safe_layer}__{safe_tensor}.pt"
    filepath = os.path.join(iter_dir, filename)

    tensor_cpu = tensor.detach().cpu()
    torch.save(tensor_cpu, filepath)

    if log_to_metrics:
        from nvdlfw_inspect.logging import MetricLogger
        metric_prefix = f"tensor_dump/{layer_name}/{tensor_name}"
        MetricLogger.log_scalar(f"{metric_prefix}/numel", tensor.numel(), iteration)
        MetricLogger.log_scalar(
            f"{metric_prefix}/size_mb",
            tensor_cpu.numel() * tensor_cpu.element_size() / 1024 / 1024,
            iteration,
        )

    if track_state:
        TENSOR_DUMP_STATE.increment_count()


def save_tensor_dump(save_dir: str, iteration: int) -> None:
    if TENSOR_DUMP_STATE.has_data():
        logger.info(
            f"[DumpTensors] Iteration {iteration}: saved {TENSOR_DUMP_STATE.tensor_count} "
            f"tensors to {save_dir}/iter_{iteration:07d}/"
        )
    TENSOR_DUMP_STATE.reset()


class TensorDumpBuffer:
    """Wrapper around TensorDumpState for compatibility."""

    def __init__(self):
        self._state = TENSOR_DUMP_STATE

    def reset(self):
        self._state.reset()

    @property
    def current_iteration(self) -> Optional[int]:
        return self._state.current_iteration

    @current_iteration.setter
    def current_iteration(self, value: Optional[int]):
        self._state.current_iteration = value

    @property
    def save_dir(self) -> Optional[str]:
        return self._state.save_dir

    @save_dir.setter
    def save_dir(self, value: Optional[str]):
        self._state.save_dir = value

    def add_tensor(self, layer_name: str, tensor_name: str, tensor: torch.Tensor):
        if self._state.save_dir and self._state.current_iteration is not None:
            save_tensor_direct(
                self._state.save_dir,
                self._state.current_iteration,
                layer_name,
                tensor_name,
                tensor,
            )

    def has_data(self) -> bool:
        return self._state.has_data()
