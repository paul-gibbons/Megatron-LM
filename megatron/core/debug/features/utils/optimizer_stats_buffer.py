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

"""Optimizer statistics buffer."""

import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import torch

from nvdlfw_inspect.logging import MetricLogger

from megatron.core.debug.features.utils.optimizer_stats_computation import (
    STAT_INDICES,
    NUM_BUFFER_STATS,
    compute_buffer_stats,
    compute_final_stats,
)

logger = logging.getLogger(__name__)


def _synchronize_param_names(
    local_names: Set[str],
    reduction_group: Optional[torch.distributed.ProcessGroup],
) -> List[str]:
    """Gather and sort parameter names across all ranks for synchronized collective ops."""
    if reduction_group is None or not torch.distributed.is_initialized():
        return sorted(local_names)

    world_size = torch.distributed.get_world_size(group=reduction_group)
    gathered_names = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered_names, local_names, group=reduction_group)

    all_names: Set[str] = set()
    for names in gathered_names:
        if names:
            all_names.update(names)

    return sorted(all_names)


def extract_layer_key(param_name: str, aggregate_by: str) -> Optional[str]:
    """Extract layer key from param name for aggregation."""
    if aggregate_by == "layer":
        match = re.search(r"decoder\.layers\.(\d+)", param_name)
        if match:
            return f"layer_{match.group(1)}"
        if "embedding" in param_name:
            return "embedding"
        if "output_layer" in param_name:
            return "output_layer"
        if "norm" in param_name.lower():
            return "layernorm"
        return "other"

    if aggregate_by == "module":
        if "embedding" in param_name:
            return "embedding"
        if "self_attention" in param_name:
            return "attention"
        if "mlp" in param_name or "expert" in param_name:
            return "mlp"
        if "output_layer" in param_name:
            return "output_layer"
        if "norm" in param_name.lower():
            return "layernorm"
        return "other"

    try:
        match = re.search(aggregate_by, param_name)
        if match:
            return match.group("key") if "key" in match.groupdict() else match.group(0)
    except re.error:
        logger.warning(f"Invalid aggregate_by regex: {aggregate_by}")
    return None


class _OptimizerStatsBuffer:
    """Buffer for one optimizer parameter."""

    def __init__(self, param_name: str, stats: List[str], reduction_group, aggregate_by: Optional[str]):
        self.param_name = param_name
        self.stats = stats
        self.reduction_group = reduction_group
        self.aggregate_by = aggregate_by

        self._buffer = torch.zeros(NUM_BUFFER_STATS, dtype=torch.float64, device="cuda")
        self.modified = torch.tensor([False], dtype=torch.bool, device="cuda")
        self.iteration: Optional[int] = None

        from megatron.core.debug.features.utils.stats_computation import parse_num_zeros_stat
        self._num_zeros: Dict[float, float] = {}
        self._num_zeros_numel = 0
        self._num_zeros_thresholds = set()
        for stat in stats:
            parsed = parse_num_zeros_stat(stat)
            if parsed:
                self._num_zeros_thresholds.add(parsed[0])

    def feed(self, param: torch.Tensor, grad: Optional[torch.Tensor],
             optimizer_state: Dict, iteration: int):
        self.iteration = iteration
        new_buffer = compute_buffer_stats(param, grad, optimizer_state, self.stats)

        if self.modified[0]:
            self._buffer += new_buffer
        else:
            self._buffer.copy_(new_buffer)

        if grad is not None and self._num_zeros_thresholds:
            abs_g = grad.float().abs()
            self._num_zeros_numel += grad.numel()
            for threshold in self._num_zeros_thresholds:
                count = (grad == 0).sum().item() if threshold == 0.0 else (abs_g < threshold).sum().item()
                self._num_zeros[threshold] = self._num_zeros.get(threshold, 0) + count

        self.modified[0] = True

    def _reduce_buffer(self) -> torch.Tensor:
        if self.reduction_group is None:
            return self._buffer
        reduced = self._buffer.clone()
        torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM, group=self.reduction_group)
        return reduced

    def _reduce_num_zeros(self) -> Tuple[Dict[float, float], int]:
        if self.reduction_group is None:
            return self._num_zeros.copy(), self._num_zeros_numel

        thresholds = sorted(self._num_zeros_thresholds)
        if not thresholds:
            return {}, 0

        data = torch.zeros(len(thresholds) + 1, dtype=torch.float64, device="cuda")
        for i, t in enumerate(thresholds):
            data[i] = self._num_zeros.get(t, 0)
        data[-1] = self._num_zeros_numel
        torch.distributed.all_reduce(data, op=torch.distributed.ReduceOp.SUM, group=self.reduction_group)

        reduced_counts = {t: data[i].item() for i, t in enumerate(thresholds)}
        reduced_numel = int(data[-1].item())
        return reduced_counts, reduced_numel

    def log(self) -> Dict[Tuple, float]:
        if not self.modified[0]:
            return {}

        output = {}
        name_prefix = extract_layer_key(self.param_name, self.aggregate_by) if self.aggregate_by else self.param_name
        reduced_buffer = self._reduce_buffer()
        reduced_num_zeros, reduced_numel = self._reduce_num_zeros()

        from megatron.core.debug.features.utils.stats_computation import parse_num_zeros_stat
        for stat in self.stats:
            parsed = parse_num_zeros_stat(stat)
            if parsed:
                threshold, is_pct = parsed
                count = reduced_num_zeros.get(threshold, 0)
                threshold_str = "" if threshold == 0.0 else f"[{threshold:g}]"
                if is_pct:
                    value = 100.0 * count / reduced_numel if reduced_numel > 0 else 0.0
                    stat_name = f"num_zeros{threshold_str}%"
                else:
                    value = count
                    stat_name = f"num_zeros{threshold_str}"
                metric_name = f"optimizer_{name_prefix}_{stat_name}"
                MetricLogger.log_scalar(metric_name, value, self.iteration)
                output[(name_prefix, stat_name, self.iteration)] = value

        final = compute_final_stats(reduced_buffer, self.stats)
        for stat_name, value in final.items():
            metric_name = f"optimizer_{name_prefix}_{stat_name}"
            MetricLogger.log_scalar(metric_name, value, self.iteration)
            output[(name_prefix, stat_name, self.iteration)] = value

        self._reset()
        return output

    def log_synchronized(
        self,
        reduction_group: Optional[torch.distributed.ProcessGroup],
    ) -> Dict[Tuple, float]:
        """Log stats with synchronized all_reduce (all ranks participate even if unmodified)."""
        output = {}
        name_prefix = (
            extract_layer_key(self.param_name, self.aggregate_by)
            if self.aggregate_by
            else self.param_name
        )

        if self.modified[0]:
            buffer_to_reduce = self._buffer.clone()
        else:
            buffer_to_reduce = torch.zeros(NUM_BUFFER_STATS, dtype=torch.float64, device="cuda")

        if reduction_group is not None:
            torch.distributed.all_reduce(
                buffer_to_reduce, op=torch.distributed.ReduceOp.SUM, group=reduction_group
            )

        thresholds = sorted(self._num_zeros_thresholds)
        reduced_num_zeros: Dict[float, float] = {}
        reduced_numel = 0

        if thresholds:
            data = torch.zeros(len(thresholds) + 1, dtype=torch.float64, device="cuda")
            if self.modified[0]:
                for i, t in enumerate(thresholds):
                    data[i] = self._num_zeros.get(t, 0)
                data[-1] = self._num_zeros_numel

            if reduction_group is not None:
                torch.distributed.all_reduce(
                    data, op=torch.distributed.ReduceOp.SUM, group=reduction_group
                )

            reduced_num_zeros = {t: data[i].item() for i, t in enumerate(thresholds)}
            reduced_numel = int(data[-1].item())
        elif self.modified[0]:
            reduced_numel = self._num_zeros_numel

        has_data = buffer_to_reduce.abs().sum() > 0 or reduced_numel > 0

        if has_data and self.iteration is not None:
            from megatron.core.debug.features.utils.stats_computation import parse_num_zeros_stat
            for stat in self.stats:
                parsed = parse_num_zeros_stat(stat)
                if parsed:
                    threshold, is_pct = parsed
                    count = reduced_num_zeros.get(threshold, 0)
                    threshold_str = "" if threshold == 0.0 else f"[{threshold:g}]"
                    if is_pct:
                        value = 100.0 * count / reduced_numel if reduced_numel > 0 else 0.0
                        stat_name = f"num_zeros{threshold_str}%"
                    else:
                        value = count
                        stat_name = f"num_zeros{threshold_str}"
                    metric_name = f"optimizer_{name_prefix}_{stat_name}"
                    MetricLogger.log_scalar(metric_name, value, self.iteration)
                    output[(name_prefix, stat_name, self.iteration)] = value

            final = compute_final_stats(buffer_to_reduce, self.stats)
            for stat_name, value in final.items():
                metric_name = f"optimizer_{name_prefix}_{stat_name}"
                MetricLogger.log_scalar(metric_name, value, self.iteration)
                output[(name_prefix, stat_name, self.iteration)] = value

        self._reset()
        return output

    def get_num_zeros_data(self) -> Tuple[Dict[float, float], int]:
        return self._num_zeros.copy(), self._num_zeros_numel

    def _reset(self):
        self._buffer.zero_()
        self.modified[0] = False
        self._num_zeros.clear()
        self._num_zeros_numel = 0


class OptimizerStatsBuffers:
    """Manager for optimizer stat buffers."""

    def __init__(self):
        self.buffers: Dict[str, _OptimizerStatsBuffer] = {}
        self.reduction_group_to_buffers: Dict = defaultdict(list)
        self.at_least_one_fed = False
        self.aggregate_by: Optional[str] = None

    def reset(self):
        self.buffers.clear()
        self.reduction_group_to_buffers.clear()
        self.at_least_one_fed = False
        self.aggregate_by = None

    def try_add_buffer(self, param_name: str, stats: List[str],
                       reduction_group, aggregate_by: Optional[str] = None):
        if param_name in self.buffers:
            return
        buffer = _OptimizerStatsBuffer(param_name, stats, reduction_group, aggregate_by)
        self.buffers[param_name] = buffer
        self.reduction_group_to_buffers[reduction_group].append(buffer)
        if aggregate_by:
            self.aggregate_by = aggregate_by

    def feed(self, param_name: str, param: torch.Tensor, grad: Optional[torch.Tensor],
             optimizer_state: Dict, stats: List[str], iteration: int,
             reduction_group=None, aggregate_by: Optional[str] = None):
        self.at_least_one_fed = True
        self.try_add_buffer(param_name, stats, reduction_group, aggregate_by)
        self.buffers[param_name].feed(param, grad, optimizer_state, iteration)

    def log_stats(self, current_iter: int) -> Dict[Tuple, float]:
        if not self.at_least_one_fed:
            return {}

        if self.aggregate_by:
            return self._log_aggregated(current_iter)

        reduction_group = None
        for buffers in self.reduction_group_to_buffers.values():
            if buffers:
                reduction_group = buffers[0].reduction_group
                break

        local_names = set(self.buffers.keys())
        all_param_names = _synchronize_param_names(local_names, reduction_group)

        output = {}
        for param_name in all_param_names:
            if param_name in self.buffers:
                buffer = self.buffers[param_name]
                output.update(buffer.log_synchronized(reduction_group))
            elif reduction_group is not None:
                self._participate_in_reduction_only(param_name, reduction_group)

        self.at_least_one_fed = False
        return output

    def _participate_in_reduction_only(
        self,
        param_name: str,
        reduction_group: torch.distributed.ProcessGroup,
    ) -> None:
        """Participate in all_reduce with zeros for a parameter this rank doesn't have."""
        sample_buffer = next(iter(self.buffers.values()), None)
        if sample_buffer is None:
            return

        zero_buffer = torch.zeros(NUM_BUFFER_STATS, dtype=torch.float64, device="cuda")
        torch.distributed.all_reduce(
            zero_buffer, op=torch.distributed.ReduceOp.SUM, group=reduction_group
        )

        thresholds = sorted(sample_buffer._num_zeros_thresholds)
        if thresholds:
            zero_data = torch.zeros(len(thresholds) + 1, dtype=torch.float64, device="cuda")
            torch.distributed.all_reduce(
                zero_data, op=torch.distributed.ReduceOp.SUM, group=reduction_group
            )

    def _log_aggregated(self, current_iter: int) -> Dict[Tuple, float]:
        from megatron.core.debug.features.utils.stats_computation import parse_num_zeros_stat

        reduction_group = None
        for buffer in self.buffers.values():
            reduction_group = buffer.reduction_group
            break

        local_grouped: Dict[str, List[_OptimizerStatsBuffer]] = defaultdict(list)
        for buffer in self.buffers.values():
            key = extract_layer_key(buffer.param_name, self.aggregate_by) or "unknown"
            local_grouped[key].append(buffer)

        local_layer_keys = set(local_grouped.keys())
        all_layer_keys = _synchronize_param_names(local_layer_keys, reduction_group)

        stats = None
        iteration = current_iter
        all_thresholds: Set[float] = set()
        for buffer in self.buffers.values():
            stats = buffer.stats
            iteration = buffer.iteration if buffer.iteration is not None else iteration
            all_thresholds.update(buffer._num_zeros_thresholds)

        if stats is None:
            self.at_least_one_fed = False
            return {}

        output = {}
        for layer_key in all_layer_keys:
            combined = torch.zeros(NUM_BUFFER_STATS, dtype=torch.float64, device="cuda")
            combined_num_zeros: Dict[float, float] = {}
            combined_num_zeros_numel = 0

            if layer_key in local_grouped:
                for buffer in local_grouped[layer_key]:
                    if buffer.modified[0]:
                        combined += buffer._buffer
                        nz_dict, nz_numel = buffer.get_num_zeros_data()
                        for threshold, count in nz_dict.items():
                            combined_num_zeros[threshold] = combined_num_zeros.get(threshold, 0) + count
                        combined_num_zeros_numel += nz_numel
                    buffer._reset()

            if reduction_group is not None:
                torch.distributed.all_reduce(
                    combined, op=torch.distributed.ReduceOp.SUM, group=reduction_group
                )

                thresholds = sorted(all_thresholds)
                if thresholds:
                    data = torch.zeros(len(thresholds) + 1, dtype=torch.float64, device="cuda")
                    for i, t in enumerate(thresholds):
                        data[i] = combined_num_zeros.get(t, 0)
                    data[-1] = combined_num_zeros_numel
                    torch.distributed.all_reduce(
                        data, op=torch.distributed.ReduceOp.SUM, group=reduction_group
                    )
                    combined_num_zeros = {t: data[i].item() for i, t in enumerate(thresholds)}
                    combined_num_zeros_numel = int(data[-1].item())

            has_data = combined.abs().sum() > 0 or combined_num_zeros_numel > 0
            if not has_data:
                continue

            for stat in stats:
                parsed = parse_num_zeros_stat(stat)
                if parsed:
                    threshold, is_pct = parsed
                    count = combined_num_zeros.get(threshold, 0)
                    threshold_str = "" if threshold == 0.0 else f"[{threshold:g}]"
                    if is_pct:
                        value = 100.0 * count / combined_num_zeros_numel if combined_num_zeros_numel > 0 else 0.0
                        stat_name = f"num_zeros{threshold_str}%"
                    else:
                        value = count
                        stat_name = f"num_zeros{threshold_str}"
                    metric_name = f"optimizer_{layer_key}_{stat_name}"
                    MetricLogger.log_scalar(metric_name, value, iteration)
                    output[(layer_key, stat_name, iteration)] = value

            final = compute_final_stats(combined, stats)
            for stat_name, value in final.items():
                metric_name = f"optimizer_{layer_key}_{stat_name}"
                MetricLogger.log_scalar(metric_name, value, iteration)
                output[(layer_key, stat_name, iteration)] = value

        self.at_least_one_fed = False
        return output


OPTIMIZER_STATS_BUFFERS = OptimizerStatsBuffers()
