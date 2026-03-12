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

"""Stats buffer for MCore tensor statistics."""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

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
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        self.layer_name = layer_name
        self.tensor_name = tensor_name
        self.reduction_group = reduction_group
        self.tp_group = tp_group
        self.reduce_within_microbatch = reduce_within_microbatch
        self.stats_to_log = stats_to_log

        self.stats_to_compute: Set[str] = set()
        for stat in stats_to_log:
            stat_lower = stat.lower()
            if stat_lower in STAT_DEPENDENCIES:
                self.stats_to_compute.update(STAT_DEPENDENCIES[stat_lower])

        self._buffer = torch.zeros(NUM_BUFFER_STATS, dtype=torch.float32, device="cuda")
        self._tmp_buffer = self._buffer.clone()
        self.modified = torch.tensor([False], dtype=torch.bool, device="cuda")
        self.iteration: Optional[int] = None
        self.skip_reduction = False

        self._direct_stats_sum: Dict[str, float] = defaultdict(float)
        self._direct_stats_count: Dict[str, int] = defaultdict(int)
        self._per_element_acc: Optional[torch.Tensor] = None
        self._num_zeros_counts: Dict[float, torch.Tensor] = {}
        self._num_zeros_numel: Optional[torch.Tensor] = None
        self._num_zeros_thresholds: Set[float] = set()

        for stat in stats_to_log:
            parsed = parse_num_zeros_stat(stat)
            if parsed:
                self._num_zeros_thresholds.add(parsed[0])

    def _reset(self):
        self._buffer.zero_()
        self.modified[0] = False
        self._direct_stats_sum.clear()
        self._direct_stats_count.clear()
        self._per_element_acc = None
        self._num_zeros_counts.clear()
        self._num_zeros_numel = None
        self._num_zeros_reduced = False

    def feed(self, tensor: torch.Tensor, iteration: int, skip_reduction: bool = False):
        self.iteration = iteration
        self.skip_reduction = skip_reduction

        if self.modified[0] and not self.reduce_within_microbatch:
            return
        if tensor.numel() == 0:
            return

        # Compute stats into tmp buffer
        for stat_name in self.stats_to_compute:
            if stat_name not in STAT_INDICES:
                continue
            compute_fn, _ = STATS[stat_name]
            self._tmp_buffer[STAT_INDICES[stat_name]] = compute_fn(tensor)

        # Accumulate using combinator (same pattern as TE)
        # Stack [old_buffer, new_buffer] and apply combinator
        if self.modified[0]:
            buffers = torch.stack([self._buffer, self._tmp_buffer], dim=0)
            for stat_name in self.stats_to_compute:
                if stat_name not in STAT_INDICES:
                    continue
                _, combinator = STATS[stat_name]
                self._buffer[STAT_INDICES[stat_name]] = combinator(buffers)
        else:
            self._buffer.copy_(self._tmp_buffer)

        for stat in self.stats_to_log:
            stat_lower = stat.lower()
            if stat_lower in DIRECT_STATS:
                self._direct_stats_sum[stat_lower] += float(DIRECT_STATS[stat_lower](tensor))
                self._direct_stats_count[stat_lower] += 1
            elif stat_lower in ("per_element", "per_element%"):
                flat = tensor.float().flatten().detach()
                if self._per_element_acc is None:
                    self._per_element_acc = flat.clone()
                elif self._per_element_acc.shape == flat.shape:
                    self._per_element_acc += flat
                else:
                    logger.warning("[MCore] Per-element shape mismatch, resetting")
                    self._per_element_acc = flat.clone()

        if self._num_zeros_thresholds:
            abs_t = tensor.float().abs()
            numel_t = torch.tensor([tensor.numel()], dtype=torch.float32, device="cuda")
            if self._num_zeros_numel is None:
                self._num_zeros_numel = numel_t
            else:
                self._num_zeros_numel += numel_t
            for threshold in self._num_zeros_thresholds:
                count = (tensor == 0).sum() if threshold == 0.0 else (abs_t < threshold).sum()
                count_t = count.float().unsqueeze(0)
                if threshold not in self._num_zeros_counts:
                    self._num_zeros_counts[threshold] = count_t
                else:
                    self._num_zeros_counts[threshold] += count_t

        self.modified[0] = True

    def _gather_buffers(self) -> torch.Tensor:
        """Gather stats from all ranks, filtering to only modified ranks (mirrors TE pattern)."""
        if self.skip_reduction or self.reduction_group is None:
            return self._buffer.unsqueeze(0)
        mask = gather_along_first_dim(self.modified, process_group=self.reduction_group)[0]
        gathered, _ = gather_along_first_dim(
            self._buffer.unsqueeze(0), process_group=self.reduction_group
        )
        return gathered[mask.to(torch.bool)]

    def log(self) -> Dict[Tuple, float]:
        gathered = self._gather_buffers()
        if not self.modified[0]:
            return {}

        output = {}

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

            if stat_lower in self._direct_stats_sum:
                count = self._direct_stats_count.get(stat_lower, 0)
                if count > 0:
                    value = self._direct_stats_sum[stat_lower] / count
            elif stat_lower in STATS:
                _, combine_fn = STATS[stat_lower]
                if combine_fn:
                    value = combine_fn(gathered)
                    # Convert tensor to Python float for logging
                    if hasattr(value, 'item'):
                        value = value.item()

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

    def _reduce_num_zeros_once(self):
        """All-reduce num_zeros counts and numel once."""
        if getattr(self, "_num_zeros_reduced", False):
            return
        if self.skip_reduction or self.reduction_group is None:
            self._num_zeros_reduced = True
            return
        if self._num_zeros_numel is not None:
            torch.distributed.all_reduce(self._num_zeros_numel, group=self.reduction_group)
        for count_t in self._num_zeros_counts.values():
            torch.distributed.all_reduce(count_t, group=self.reduction_group)
        self._num_zeros_reduced = True

    def _log_num_zeros(self, stat: str, parsed: Tuple[float, bool], output: dict):
        threshold, is_pct = parsed
        count_t = self._num_zeros_counts.get(threshold)
        if count_t is None or self._num_zeros_numel is None:
            return

        # Reduce all num_zeros counts once
        self._reduce_num_zeros_once()

        count = count_t.item()
        numel = self._num_zeros_numel.item()

        threshold_str = "" if threshold == 0.0 else f"[{threshold:g}]"
        if is_pct:
            value = (count / numel * 100) if numel > 0 else 0.0
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
                       options: tuple, reduction_group, reduce_within_microbatch: bool = True,
                       tp_group=None):
        key = (layer_name, tensor_name, options)
        if key in self.buffers:
            return
        buffer = _MCoreStatsBuffer(
            layer_name, tensor_name, stats, reduction_group, reduce_within_microbatch, tp_group
        )
        self.buffers[key] = buffer
        self.reduction_group_to_buffers[reduction_group].append(buffer)

    def feed(self, layer_name: str, tensor_name: str, options: tuple,
             tensor: torch.Tensor, iteration: int, skip_reduction: bool = False):
        self.at_least_one_fed = True
        self.buffers[(layer_name, tensor_name, options)].feed(tensor, iteration, skip_reduction)

    def _should_run_reduction(self) -> bool:
        """Check if reduction should be run (mirrors TE's _if_run_reduction pattern)."""
        if self.at_least_one_fed:
            return True
        from megatron.core.debug.debug_state import MCoreDebugState
        iteration = MCoreDebugState.get_iteration()
        layers_to_remove = []
        for layer, next_iter in self.layers_to_next_iter.items():
            if next_iter is None:
                layers_to_remove.append(layer)
                continue
            if iteration >= next_iter:
                return True
        for layer in layers_to_remove:
            self.layers_to_next_iter.pop(layer, None)
        return False

    def _any_rank_should_run_reduction(self, should_run_local: bool) -> bool:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return should_run_local

        device = "cuda" if torch.cuda.is_available() else "cpu"
        flag = torch.tensor([int(should_run_local)], dtype=torch.int32, device=device)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.SUM)
        return bool(flag.item())

    def log_stats(self) -> Dict[Tuple, float]:
        """Log stats from all buffers. Called by MegatronCoreAPI.step()."""
        should_run_local = self._should_run_reduction()
        if not self._any_rank_should_run_reduction(should_run_local):
            self.at_least_one_fed = False
            return {}

        output = {}
        for reduction_group, buffers in self.reduction_group_to_buffers.items():
            for buffer in buffers:
                if buffer.skip_reduction or reduction_group is None:
                    should_log = bool(buffer.modified[0].item())
                else:
                    changed_mask, _ = gather_along_first_dim(
                        buffer.modified.unsqueeze(0), process_group=reduction_group
                    )
                    should_log = bool(changed_mask.any().item())

                if should_log:
                    output.update(buffer.log())
        self.at_least_one_fed = False
        return output


MCORE_STATS_BUFFERS = MCoreStatsBuffers()
