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

import math
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch

from nvdlfw_inspect.logging import MetricLogger
from nvdlfw_inspect.utils import gather_along_first_dim


STAT_INDICES = {
    "min": 0,
    "max": 1,
    "sum": 2,
    "numel": 3,
    "sum_sq": 4,
    "l1_norm": 5,
    "l2_norm_sq": 6,
    "cur_amax": 7,
    "dynamic_range_top": 8,
    "dynamic_range_bottom": 9,
    "sparsity_zeros": 10,
}

NUM_BUFFER_STATS = 11


STAT_DEPENDENCIES: Dict[str, Set[str]] = {
    "min": {"min"},
    "max": {"max"},
    "sum": {"sum"},
    "mean": {"sum", "numel"},
    "numel": {"numel"},
    "std": {"sum", "numel", "sum_sq"},
    "variance": {"sum", "numel", "sum_sq"},
    "l1_norm": {"l1_norm"},
    "l2_norm": {"l2_norm_sq"},
    "cur_amax": {"cur_amax"},
    "dynamic_range": {"dynamic_range_top", "dynamic_range_bottom"},
    "sparsity": {"sparsity_zeros", "numel"},
}


def _compute_base_stats(tensor: torch.Tensor, stats_needed: Set[str]) -> Dict[str, torch.Tensor]:
    results = {}
    t_float = tensor.float()

    if "min" in stats_needed:
        results["min"] = t_float.min()
    if "max" in stats_needed:
        results["max"] = t_float.max()
    if "sum" in stats_needed:
        results["sum"] = t_float.sum()
    if "numel" in stats_needed:
        results["numel"] = torch.tensor(tensor.numel(), dtype=torch.float32, device=tensor.device)
    if "sum_sq" in stats_needed:
        results["sum_sq"] = (t_float ** 2).sum()
    if "l1_norm" in stats_needed:
        results["l1_norm"] = torch.norm(t_float, p=1)
    if "l2_norm_sq" in stats_needed:
        results["l2_norm_sq"] = (t_float ** 2).sum()
    if "cur_amax" in stats_needed:
        results["cur_amax"] = t_float.abs().max()
    if "dynamic_range_top" in stats_needed or "dynamic_range_bottom" in stats_needed:
        abs_t = t_float.abs()
        nonzero_mask = abs_t > 0
        if nonzero_mask.any():
            results["dynamic_range_top"] = torch.log2(abs_t.max())
            results["dynamic_range_bottom"] = torch.log2(abs_t[nonzero_mask].min())
        else:
            results["dynamic_range_top"] = torch.tensor(0.0, device=tensor.device)
            results["dynamic_range_bottom"] = torch.tensor(0.0, device=tensor.device)
    if "sparsity_zeros" in stats_needed:
        results["sparsity_zeros"] = (tensor == 0).sum().float()
    
    return results


def _combine_stats(buffers: torch.Tensor, stat_name: str) -> float:
    def get(name: str) -> torch.Tensor:
        return buffers[:, STAT_INDICES[name]]
    
    if stat_name == "min":
        return get("min").min().item()
    elif stat_name == "max":
        return get("max").max().item()
    elif stat_name == "sum":
        return get("sum").sum().item()
    elif stat_name == "numel":
        return get("numel").sum().item()
    elif stat_name == "mean":
        total_sum = get("sum").sum()
        total_numel = get("numel").sum()
        return (total_sum / total_numel).item()
    elif stat_name == "variance":
        total_numel = get("numel").sum()
        total_sum = get("sum").sum()
        total_sum_sq = get("sum_sq").sum()
        mean = total_sum / total_numel
        return ((total_sum_sq / total_numel) - mean ** 2).item()
    elif stat_name == "std":
        total_numel = get("numel").sum()
        total_sum = get("sum").sum()
        total_sum_sq = get("sum_sq").sum()
        mean = total_sum / total_numel
        variance = (total_sum_sq / total_numel) - mean ** 2
        return torch.sqrt(variance.clamp(min=0)).item()
    elif stat_name == "l1_norm":
        return get("l1_norm").sum().item()
    elif stat_name == "l2_norm":
        return math.sqrt(get("l2_norm_sq").sum().item())
    elif stat_name == "cur_amax":
        return get("cur_amax").max().item()
    elif stat_name == "dynamic_range":
        top = get("dynamic_range_top").max()
        bottom = get("dynamic_range_bottom").min()
        return (top - bottom).item()
    elif stat_name == "sparsity":
        total_zeros = get("sparsity_zeros").sum()
        total_numel = get("numel").sum()
        return (total_zeros / total_numel).item()
    else:
        raise ValueError(f"Unknown stat: {stat_name}")


class _MCoreStatsBuffer:
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
        self._tmp_buffer = torch.zeros(NUM_BUFFER_STATS, dtype=torch.float32, device="cuda")
        self.modified = torch.tensor([False], dtype=torch.bool, device="cuda")
        self.iteration: Optional[int] = None
        self.skip_reduction = False
        self._direct_stats: Dict[str, float] = {}
        self._per_element_values: Optional[torch.Tensor] = None
        self._per_element_accumulated: Optional[torch.Tensor] = None
        self._per_element_count: int = 0

    def _reset(self):
        self._buffer.zero_()
        self.modified[0] = False
        self._direct_stats.clear()
        self._per_element_values = None
        self._per_element_accumulated = None
        self._per_element_count = 0

    def feed(
        self,
        tensor: torch.Tensor,
        iteration: int,
        skip_reduction: bool = False,
    ):
        self.iteration = iteration
        self.skip_reduction = skip_reduction

        if self.modified[0] and not self.reduce_within_microbatch:
            return
        
        if tensor.numel() == 0:
            return

        base_stats = _compute_base_stats(tensor, self.stats_to_compute)

        for stat_name, value in base_stats.items():
            if stat_name in STAT_INDICES:
                self._tmp_buffer[STAT_INDICES[stat_name]] = value

        if self.modified[0]:
            for stat_name in self.stats_to_compute:
                if stat_name not in STAT_INDICES:
                    continue
                idx = STAT_INDICES[stat_name]
                old_val = self._buffer[idx]
                new_val = self._tmp_buffer[idx]

                if stat_name == "min":
                    self._buffer[idx] = torch.min(old_val, new_val)
                elif stat_name == "max":
                    self._buffer[idx] = torch.max(old_val, new_val)
                elif stat_name in ("sum", "numel", "sum_sq", "l1_norm", "l2_norm_sq", "sparsity_zeros"):
                    self._buffer[idx] = old_val + new_val
                elif stat_name == "cur_amax":
                    self._buffer[idx] = torch.max(old_val, new_val)
                elif stat_name == "dynamic_range_top":
                    self._buffer[idx] = torch.max(old_val, new_val)
                elif stat_name == "dynamic_range_bottom":
                    self._buffer[idx] = torch.min(old_val, new_val)
        else:
            self._buffer.copy_(self._tmp_buffer)

        for stat in self.stats_to_log:
            stat_lower = stat.lower()
            if stat_lower in ("per_element", "per_element%"):
                flat = tensor.float().flatten().detach()
                if self._per_element_accumulated is None:
                    self._per_element_accumulated = flat.clone()
                    self._per_element_count = 1
                elif self._per_element_accumulated.shape == flat.shape:
                    self._per_element_accumulated += flat
                    self._per_element_count += 1
                else:
                    self._per_element_accumulated = flat.clone()
                    self._per_element_count = 1
            elif stat_lower == "entropy":
                flat = tensor.float().flatten()
                p = flat.clamp(min=1e-10)
                if p.sum() > 0:
                    p = p / p.sum()
                entropy = -(p * torch.log(p)).sum()
                self._direct_stats["entropy"] = entropy.item()
            elif stat_lower in ("median", "q1", "q3", "iqr", "max_median_ratio"):
                flat = tensor.float().flatten()
                if stat_lower == "median":
                    self._direct_stats["median"] = flat.median().item()
                elif stat_lower == "q1":
                    self._direct_stats["q1"] = torch.quantile(flat, 0.25).item()
                elif stat_lower == "q3":
                    self._direct_stats["q3"] = torch.quantile(flat, 0.75).item()
                elif stat_lower == "iqr":
                    q1 = torch.quantile(flat, 0.25)
                    q3 = torch.quantile(flat, 0.75)
                    self._direct_stats["iqr"] = (q3 - q1).item()
                elif stat_lower == "max_median_ratio":
                    median_val = flat.median()
                    max_val = flat.max()
                    if median_val != 0:
                        self._direct_stats["max_median_ratio"] = (max_val / median_val).item()
                    else:
                        self._direct_stats["max_median_ratio"] = float('inf')

        self.modified[0] = True

    def _gather_buffers(self) -> torch.Tensor:
        if self.skip_reduction or self.reduction_group is None:
            return self._buffer.unsqueeze(0)

        mask, _ = gather_along_first_dim(
            self.modified.unsqueeze(0), process_group=self.reduction_group
        )
        gathered, _ = gather_along_first_dim(
            self._buffer.unsqueeze(0), process_group=self.reduction_group
        )
        mask_1d = mask.flatten().bool()
        return gathered[mask_1d]

    def log(self) -> Dict[Tuple, float]:
        if not self.modified[0]:
            return {}

        output = {}
        gathered_buffers = self._gather_buffers()

        for stat in self.stats_to_log:
            stat_lower = stat.lower()
            
            if stat_lower in ("per_element", "per_element%"):
                if self._per_element_accumulated is not None:
                    if not self.skip_reduction and self.reduction_group is not None:
                        try:
                            gathered, _ = gather_along_first_dim(
                                self._per_element_accumulated.unsqueeze(0),
                                process_group=self.reduction_group
                            )
                            per_element_total = gathered.sum(dim=0)
                        except Exception:
                            per_element_total = self._per_element_accumulated
                    else:
                        per_element_total = self._per_element_accumulated

                    total_tokens = per_element_total.sum()
                    for idx, val in enumerate(per_element_total):
                        if stat_lower == "per_element":
                            metric_name = f"{self.layer_name}_{self.tensor_name}_expert{idx}"
                            MetricLogger.log_scalar(metric_name, val.item(), self.iteration)
                            output[(self.layer_name, self.tensor_name, f"expert_{idx}", self.iteration)] = val.item()
                        else:
                            if total_tokens > 0:
                                pct = (val / total_tokens * 100).item()
                                metric_name = f"{self.layer_name}_{self.tensor_name}_expert{idx}%"
                                MetricLogger.log_scalar(metric_name, pct, self.iteration)
                                output[(self.layer_name, self.tensor_name, f"expert_{idx}%", self.iteration)] = pct
                continue

            if stat_lower in self._direct_stats:
                value = self._direct_stats[stat_lower]
            elif stat_lower in STAT_DEPENDENCIES:
                value = _combine_stats(gathered_buffers, stat_lower)
            else:
                continue

            metric_name = f"{self.layer_name}_{self.tensor_name}_{stat_lower}"
            MetricLogger.log_scalar(metric_name, value, self.iteration)
            output[(self.layer_name, self.tensor_name, stat_lower, self.iteration)] = value

        self._reset()
        return output


class MCoreStatsBuffers:
    def __init__(self):
        self.buffers: Dict[Tuple[str, str, tuple], _MCoreStatsBuffer] = {}
        self.reduction_group_to_buffers: Dict[Optional[torch.distributed.ProcessGroup], List[_MCoreStatsBuffer]] = defaultdict(list)
        self.at_least_one_fed = False
        self.layers_to_next_iter: Dict[str, Optional[int]] = {}

    def reset(self):
        self.buffers.clear()
        self.reduction_group_to_buffers.clear()
        self.at_least_one_fed = False
        self.layers_to_next_iter.clear()

    def _should_run_reduction(self, current_iter: int) -> bool:
        if self.at_least_one_fed:
            return True

        layers_to_remove = []
        for layer_name, next_iter in self.layers_to_next_iter.items():
            if next_iter is None:
                layers_to_remove.append(layer_name)
                continue
            if current_iter >= next_iter:
                return True

        for layer_name in layers_to_remove:
            self.layers_to_next_iter.pop(layer_name, None)

        return False

    def try_add_buffer(
        self,
        layer_name: str,
        tensor_name: str,
        stats: List[str],
        options: tuple,
        reduction_group: Optional[torch.distributed.ProcessGroup],
        reduce_within_microbatch: bool = True,
    ):
        key = (layer_name, tensor_name, options)
        if key in self.buffers:
            return
        
        buffer = _MCoreStatsBuffer(
            layer_name=layer_name,
            tensor_name=tensor_name,
            stats_to_log=stats,
            reduction_group=reduction_group,
            reduce_within_microbatch=reduce_within_microbatch,
        )
        self.buffers[key] = buffer
        self.reduction_group_to_buffers[reduction_group].append(buffer)

    def feed(
        self,
        layer_name: str,
        tensor_name: str,
        options: tuple,
        tensor: torch.Tensor,
        iteration: int,
        skip_reduction: bool = False,
    ):
        self.at_least_one_fed = True
        key = (layer_name, tensor_name, options)
        buffer = self.buffers[key]
        buffer.feed(tensor, iteration, skip_reduction)

    def log_stats(self, current_iter: int) -> Dict[Tuple, float]:
        if not self._should_run_reduction(current_iter):
            return {}

        output = {}
        for reduction_group, buffers in self.reduction_group_to_buffers.items():
            for buffer in buffers:
                if buffer.modified[0]:
                    stats = buffer.log()
                    output.update(stats)

        self.at_least_one_fed = False
        return output


MCORE_STATS_BUFFERS = MCoreStatsBuffers()

