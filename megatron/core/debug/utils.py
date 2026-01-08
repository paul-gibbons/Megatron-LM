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

"""Utility functions for MCore debug/inspection framework."""

from typing import Optional, Tuple

import torch

from megatron.core.debug.debug_state import MCoreDebugState


class TensorInspectMixin:
    _debug_name: Optional[str] = None
    _next_debug_iter: Optional[int] = 0
    _debug_last_iteration: Optional[int] = None
    _debug_enabled_this_iter: bool = False

    def _get_debug_name(self) -> str:
        raise NotImplementedError

    def _get_reduction_group(self) -> Optional[torch.distributed.ProcessGroup]:
        return None

    def _is_debug_iter(self) -> bool:
        MCoreDebugState.ensure_initialized()
        if not MCoreDebugState.debug_enabled:
            return False
        current_iter = MCoreDebugState.get_iteration()
        if self._debug_last_iteration != current_iter:
            self._debug_enabled_this_iter = (
                self._next_debug_iter is not None
                and current_iter >= self._next_debug_iter
            )
            self._debug_last_iteration = current_iter
        return self._debug_enabled_this_iter

    def _inspect_tensor(self, tensor_name: str, tensor: torch.Tensor) -> None:
        if not self._is_debug_iter():
            return
        import nvdlfw_inspect.api as debug_api

        layer_name = self._get_debug_name()
        iteration = MCoreDebugState.get_iteration()

        result = debug_api.megatron_core.inspect_tensor_enabled(
            layer_name=layer_name, tensor_name=tensor_name, iteration=iteration
        )

        if isinstance(result, tuple):
            enabled, next_iter = result
            if next_iter is not None:
                if self._next_debug_iter is None:
                    self._next_debug_iter = next_iter
                else:
                    self._next_debug_iter = min(self._next_debug_iter, next_iter)
        else:
            enabled = result

        if enabled:
            debug_api.megatron_core.inspect_tensor(
                layer_name=layer_name,
                tensor_name=tensor_name,
                tensor=tensor,
                iteration=iteration,
                reduction_group=self._get_reduction_group(),
            )


def is_debug_enabled() -> bool:
    MCoreDebugState.ensure_initialized()
    return MCoreDebugState.debug_enabled or False


def get_debug_iteration() -> int:
    return MCoreDebugState.get_iteration()


def compute_next_enabled_iter(
    start_step: Optional[int],
    end_step: Optional[int],
    start_end_list: Optional[list],
    freq: int,
    current_iter: int,
) -> Tuple[bool, Optional[int]]:
    start = start_step if start_step is not None else 0
    end = end_step if end_step is not None and end_step >= 0 else float("inf")

    if start_end_list is not None:
        for range_start, range_end in start_end_list:
            if range_start <= current_iter <= range_end:
                if current_iter % freq == 0:
                    next_iter = current_iter + freq
                    if next_iter > range_end:
                        for ns, ne in start_end_list:
                            if ns > current_iter:
                                return True, ns
                        return True, None
                    return True, next_iter
                else:
                    next_freq = ((current_iter // freq) + 1) * freq
                    if next_freq <= range_end:
                        return False, next_freq
                    for ns, ne in start_end_list:
                        if ns > current_iter:
                            return False, ns
                    return False, None

        for ns, ne in start_end_list:
            if ns > current_iter:
                return False, ns
        return False, None

    if current_iter < start:
        return False, start
    if current_iter > end:
        return False, None

    if current_iter % freq == 0:
        next_iter = current_iter + freq
        if next_iter > end:
            return True, None
        return True, next_iter
    else:
        next_freq = ((current_iter // freq) + 1) * freq
        if next_freq > end:
            return False, None
        return False, next_freq

