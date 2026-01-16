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
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from nvdlfw_inspect.logging import MetricLogger
from nvdlfw_inspect.utils import gather_along_first_dim

from megatron.core.debug.features.utils.stats_computation import (
    STATS, STAT_INDICES, STAT_DEPENDENCIES, DIRECT_STATS, NUM_BUFFER_STATS,
    parse_num_zeros_stat, parse_vocab_topk_stat,
    is_vocab_stat, compute_per_row_l2_norms, compute_vocab_topk_l2_pct,
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
        self.tp_group = tp_group  # For vocab stats: TP gather, then DP reduce
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

        self._direct_stats: Dict[str, float] = {}
        self._per_element_acc: Optional[torch.Tensor] = None
        self._num_zeros_counts: Dict[float, torch.Tensor] = {}
        self._num_zeros_numel: Optional[torch.Tensor] = None
        self._num_zeros_thresholds: Set[float] = set()
        self._vocab_stats_requested = False
        self._vocab_accumulated_stats: Optional[Dict] = None

        for stat in stats_to_log:
            parsed = parse_num_zeros_stat(stat)
            if parsed:
                self._num_zeros_thresholds.add(parsed[0])
            if is_vocab_stat(stat):
                self._vocab_stats_requested = True

    def _reset(self):
        self._buffer.zero_()
        self.modified[0] = False
        self._direct_stats.clear()
        self._per_element_acc = None
        self._num_zeros_counts.clear()
        self._num_zeros_numel = None
        self._num_zeros_reduced = False
        self._vocab_accumulated_stats = None
        self._vocab_finalized_stats = None

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
                self._direct_stats[stat_lower] = DIRECT_STATS[stat_lower](tensor)
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

        if self._vocab_stats_requested and tensor.dim() == 2:
            per_row_norms = compute_per_row_l2_norms(tensor)
            per_row_norms_sq = per_row_norms ** 2
            if self._vocab_accumulated_stats is None:
                self._vocab_accumulated_stats = {
                    "per_row_norms_sq_sum": per_row_norms_sq.clone(),
                    "vocab_size": tensor.shape[0],
                }
            else:
                self._vocab_accumulated_stats["per_row_norms_sq_sum"] += per_row_norms_sq

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

            # Handle vocab stats
            if is_vocab_stat(stat):
                self._log_vocab_stat(stat, output)
                continue

            if stat_lower in self._direct_stats:
                value = self._direct_stats[stat_lower]
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

    def _finalize_vocab_stats_once(self):
        """Reduce and finalize vocab stats once."""
        if getattr(self, "_vocab_finalized_stats", None) is not None:
            return self._vocab_finalized_stats
        if self._vocab_accumulated_stats is None:
            return None

        per_row_norms_sq = self._vocab_accumulated_stats["per_row_norms_sq_sum"].clone()

        if not self.skip_reduction:
            if self.tp_group is not None:
                per_row_norms_sq, _ = gather_along_first_dim(
                    per_row_norms_sq, process_group=self.tp_group
                )
            if self.reduction_group is not None and self.reduction_group != self.tp_group:
                torch.distributed.all_reduce(per_row_norms_sq, group=self.reduction_group)
        
        per_row_norms = torch.sqrt(per_row_norms_sq)
        sorted_norms, sorted_indices = torch.sort(per_row_norms, descending=True)
        total_l2_norm_sq = per_row_norms_sq.sum().item()

        self._vocab_finalized_stats = {
            "per_row_norms": per_row_norms,
            "sorted_norms": sorted_norms,
            "sorted_indices": sorted_indices,
            "total_l2_norm_sq": total_l2_norm_sq,
            "total_l2_norm": math.sqrt(total_l2_norm_sq) if total_l2_norm_sq > 0 else 0.0,
            "vocab_size": per_row_norms.shape[0],
        }
        return self._vocab_finalized_stats

    def _log_vocab_stat(self, stat: str, output: dict):
        """Log vocab_topk_l2_pct[k] statistics."""
        finalized_stats = self._finalize_vocab_stats_once()
        if finalized_stats is None:
            return

        topk = parse_vocab_topk_stat(stat)
        if topk is None:
            return

        value = compute_vocab_topk_l2_pct(finalized_stats, topk)
        stat_name = f"vocab_topk_l2_pct[{topk}]"
        metric_name = f"{self.layer_name}_{self.tensor_name}_{stat_name}"
        MetricLogger.log_scalar(metric_name, value, self.iteration)
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
