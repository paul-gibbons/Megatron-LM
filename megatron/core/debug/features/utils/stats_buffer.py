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

"""Stats buffer system for MCore tensor statistics."""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import torch

from nvdlfw_inspect.logging import MetricLogger
from nvdlfw_inspect.utils import gather_along_first_dim

from megatron.core.debug.features.utils.stats_computation import (
    STATS, STAT_INDICES, STAT_DEPENDENCIES, DIRECT_STATS, NUM_BUFFER_STATS,
    parse_num_zeros_stat,
)

logger = logging.getLogger(__name__)


class _MCoreStatsBuffer:
    """Buffer for accumulating tensor statistics across micro-batches."""

    def __init__(
        self,
        layer_name: str,
        tensor_name: str,
        stats_to_log: List[str],
        reduction_group: Optional[torch.distributed.ProcessGroup],
        reduce_within_microbatch: bool = True,
    ):
        self.layer_name = layer_name
        self.tensor_name = tensor_name
        self.reduction_group = reduction_group
        self.reduce_within_microbatch = reduce_within_microbatch
        self.stats_to_log = stats_to_log

        self.stats_to_compute: Set[str] = set()
        for stat in stats_to_log:
            stat_lower = stat.lower()
            if stat_lower in STAT_DEPENDENCIES:
                self.stats_to_compute.update(STAT_DEPENDENCIES[stat_lower])

        self._buffer = torch.zeros(NUM_BUFFER_STATS, dtype=torch.float32, device="cuda")
        self.modified = torch.tensor([False], dtype=torch.bool, device="cuda")
        self.iteration: Optional[int] = None
        self.skip_reduction = False

        self._direct_stats: Dict[str, float] = {}
        self._per_element_acc: Optional[torch.Tensor] = None
        self._num_zeros: Dict[float, int] = {}
        self._num_zeros_numel = 0
        self._num_zeros_thresholds = set()
        for stat in stats_to_log:
            parsed = parse_num_zeros_stat(stat)
            if parsed:
                self._num_zeros_thresholds.add(parsed[0])

    def _reset(self):
        self._buffer.zero_()
        self.modified[0] = False
        self._direct_stats.clear()
        self._per_element_acc = None
        self._num_zeros.clear()
        self._num_zeros_numel = 0

    def feed(self, tensor: torch.Tensor, iteration: int, skip_reduction: bool = False):
        self.iteration = iteration
        self.skip_reduction = skip_reduction

        if self.modified[0] and not self.reduce_within_microbatch:
            return
        if tensor.numel() == 0:
            return

        for stat_name in self.stats_to_compute:
            if stat_name not in STAT_INDICES:
                continue
            idx = STAT_INDICES[stat_name]
            compute_fn, _, accum_fn = STATS[stat_name]
            new_val = compute_fn(tensor)
            
            if self.modified[0] and accum_fn:
                self._buffer[idx] = accum_fn(self._buffer[idx], new_val)
            else:
                self._buffer[idx] = new_val

        for stat in self.stats_to_log:
            stat_lower = stat.lower()
            if stat_lower in DIRECT_STATS:
                self._direct_stats[stat_lower] = DIRECT_STATS[stat_lower](tensor)
            elif stat_lower in ("per_element", "per_element%"):
                flat = tensor.float().flatten().detach()
                if self._per_element_acc is None:
                    self._per_element_acc = flat.clone()
                elif self._per_element_acc.shape == flat.shape:
                    self._per_element_acc += flat
                else:
                    logger.warning(f"[MCore] Per-element shape mismatch, resetting")
                    self._per_element_acc = flat.clone()

        if self._num_zeros_thresholds:
            abs_t = tensor.float().abs()
            self._num_zeros_numel += tensor.numel()
            for threshold in self._num_zeros_thresholds:
                count = (tensor == 0).sum().item() if threshold == 0.0 else (abs_t < threshold).sum().item()
                self._num_zeros[threshold] = self._num_zeros.get(threshold, 0) + count

        self.modified[0] = True

    def _gather_buffers(self) -> torch.Tensor:
        if self.skip_reduction or self.reduction_group is None:
            return self._buffer.unsqueeze(0)
        mask, _ = gather_along_first_dim(self.modified.unsqueeze(0), process_group=self.reduction_group)
        gathered, _ = gather_along_first_dim(self._buffer.unsqueeze(0), process_group=self.reduction_group)
        return gathered[mask.flatten().bool()]

    def log(self) -> Dict[Tuple, float]:
        if not self.modified[0]:
            return {}

        output = {}
        gathered = self._gather_buffers()

        for stat in self.stats_to_log:
            stat_lower = stat.lower()
            value = None

            if stat_lower in ("per_element", "per_element%"):
                if self._per_element_acc is not None:
                    self._log_per_element(stat_lower, output)
                continue

            parsed = parse_num_zeros_stat(stat)
            if parsed:
                self._log_num_zeros(stat, parsed, output)
                continue

            if stat_lower in self._direct_stats:
                value = self._direct_stats[stat_lower]
            elif stat_lower in STATS:
                _, combine_fn, _ = STATS[stat_lower]
                if combine_fn:
                    value = combine_fn(gathered)

            if value is not None:
                metric_name = f"{self.layer_name}_{self.tensor_name}_{stat_lower}"
                MetricLogger.log_scalar(metric_name, value, self.iteration)
                output[(self.layer_name, self.tensor_name, stat_lower, self.iteration)] = value

        self._reset()
        return output

    def _log_per_element(self, stat_lower: str, output: dict):
        if self._per_element_acc is None:
            return
        
        if not self.skip_reduction and self.reduction_group is not None:
            gathered, _ = gather_along_first_dim(
                self._per_element_acc.unsqueeze(0), process_group=self.reduction_group
            )
            totals = gathered.sum(dim=0)
        else:
            totals = self._per_element_acc

        total_sum = totals.sum()
        for idx, val in enumerate(totals):
            if stat_lower == "per_element":
                key = f"expert_{idx}"
                MetricLogger.log_scalar(f"{self.layer_name}_{self.tensor_name}_expert{idx}", val.item(), self.iteration)
            else:
                key = f"expert_{idx}%"
                pct = (val / total_sum * 100).item() if total_sum > 0 else 0.0
                MetricLogger.log_scalar(f"{self.layer_name}_{self.tensor_name}_expert{idx}%", pct, self.iteration)
                val = pct
            output[(self.layer_name, self.tensor_name, key, self.iteration)] = val.item() if isinstance(val, torch.Tensor) else val

    def _log_num_zeros(self, stat: str, parsed: Tuple[float, bool], output: dict):
        threshold, is_pct = parsed
        count = self._num_zeros.get(threshold, 0)
        
        threshold_str = "" if threshold == 0.0 else f"[{threshold:g}]"
        if is_pct:
            value = (count / self._num_zeros_numel * 100) if self._num_zeros_numel > 0 else 0.0
            stat_name = f"num_zeros{threshold_str}%"
        else:
            value = count
            stat_name = f"num_zeros{threshold_str}"

        MetricLogger.log_scalar(f"{self.layer_name}_{self.tensor_name}_{stat_name}", value, self.iteration)
        output[(self.layer_name, self.tensor_name, stat_name, self.iteration)] = value


class MCoreStatsBuffers:
    """Manager for all stat buffers."""

    def __init__(self):
        self.buffers: Dict[Tuple, _MCoreStatsBuffer] = {}
        self.reduction_group_to_buffers: Dict = defaultdict(list)
        self.at_least_one_fed = False
        self.layers_to_next_iter: Dict[str, Optional[int]] = {}

    def reset(self):
        self.buffers.clear()
        self.reduction_group_to_buffers.clear()
        self.at_least_one_fed = False
        self.layers_to_next_iter.clear()

    def try_add_buffer(self, layer_name: str, tensor_name: str, stats: List[str],
                       options: tuple, reduction_group, reduce_within_microbatch: bool = True):
        key = (layer_name, tensor_name, options)
        if key in self.buffers:
            return
        buffer = _MCoreStatsBuffer(layer_name, tensor_name, stats, reduction_group, reduce_within_microbatch)
        self.buffers[key] = buffer
        self.reduction_group_to_buffers[reduction_group].append(buffer)

    def feed(self, layer_name: str, tensor_name: str, options: tuple,
             tensor: torch.Tensor, iteration: int, skip_reduction: bool = False):
        self.at_least_one_fed = True
        self.buffers[(layer_name, tensor_name, options)].feed(tensor, iteration, skip_reduction)

    def _should_run_reduction(self, current_iter: int) -> bool:
        if self.at_least_one_fed:
            return True
        for layer, next_iter in list(self.layers_to_next_iter.items()):
            if next_iter is None:
                del self.layers_to_next_iter[layer]
            elif current_iter >= next_iter:
                return True
        return False

    def log_stats(self, current_iter: int) -> Dict[Tuple, float]:
        if not self._should_run_reduction(current_iter):
            return {}
        output = {}
        for buffers in self.reduction_group_to_buffers.values():
            for buffer in buffers:
                if buffer.modified[0]:
                    output.update(buffer.log())
        self.at_least_one_fed = False
        return output


MCORE_STATS_BUFFERS = MCoreStatsBuffers()
