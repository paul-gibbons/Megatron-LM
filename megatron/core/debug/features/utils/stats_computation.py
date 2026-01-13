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

"""Stat computation functions for MCore tensor statistics."""

import math
import re
from typing import Dict, Optional, Set, Tuple

import torch

_NUM_ZEROS_PATTERN = re.compile(r'^num_zeros(?:\[([^\]]+)\])?(%)?$', re.IGNORECASE)


def parse_num_zeros_stat(stat: str) -> Optional[Tuple[float, bool]]:
    match = _NUM_ZEROS_PATTERN.match(stat.strip())
    if not match:
        return None
    threshold_str, pct_suffix = match.groups()
    return (float(threshold_str) if threshold_str else 0.0, pct_suffix is not None)


def _get(buffers: torch.Tensor, idx: int) -> torch.Tensor:
    return buffers[:, idx]


STAT_INDICES = {
    "min": 0, "max": 1, "sum": 2, "numel": 3, "sum_sq": 4,
    "l1_norm": 5, "l2_norm_sq": 6, "cur_amax": 7,
    "dynamic_range_top": 8, "dynamic_range_bottom": 9,
}
NUM_BUFFER_STATS = len(STAT_INDICES)

STAT_DEPENDENCIES: Dict[str, Set[str]] = {
    "min": {"min"}, "max": {"max"}, "sum": {"sum"}, "numel": {"numel"},
    "mean": {"sum", "numel"},
    "std": {"sum", "numel", "sum_sq"},
    "variance": {"sum", "numel", "sum_sq"},
    "l1_norm": {"l1_norm"},
    "l2_norm": {"l2_norm_sq"},
    "cur_amax": {"cur_amax"},
    "dynamic_range": {"dynamic_range_top", "dynamic_range_bottom"},
}

# STATS: (compute_fn, combinator)
# - compute_fn: tensor -> scalar tensor for buffer
# - combinator: [N, num_stats] buffer tensor -> combined value (tensor or scalar)
# The combinator handles BOTH micro-batch accumulation AND cross-rank reduction
# (same pattern as TransformerEngine)
STATS = {
    "min": (
        lambda t: t.float().min(),
        lambda b: _get(b, STAT_INDICES["min"]).min(),
    ),
    "max": (
        lambda t: t.float().max(),
        lambda b: _get(b, STAT_INDICES["max"]).max(),
    ),
    "sum": (
        lambda t: t.float().sum(),
        lambda b: _get(b, STAT_INDICES["sum"]).sum(),
    ),
    "numel": (
        lambda t: torch.tensor(float(t.numel()), device=t.device),
        lambda b: _get(b, STAT_INDICES["numel"]).sum(),
    ),
    "sum_sq": (
        lambda t: (t.float() ** 2).sum(),
        lambda b: _get(b, STAT_INDICES["sum_sq"]).sum(),
    ),
    "l1_norm": (
        lambda t: t.float().abs().sum(),
        lambda b: _get(b, STAT_INDICES["l1_norm"]).sum(),
    ),
    "l2_norm_sq": (
        lambda t: (t.float() ** 2).sum(),
        lambda b: _get(b, STAT_INDICES["l2_norm_sq"]).sum(),
    ),
    "cur_amax": (
        lambda t: t.float().abs().max(),
        lambda b: _get(b, STAT_INDICES["cur_amax"]).max(),
    ),
    "dynamic_range_top": (
        lambda t: _compute_dr_top(t),
        lambda b: _get(b, STAT_INDICES["dynamic_range_top"]).max(),
    ),
    "dynamic_range_bottom": (
        lambda t: _compute_dr_bottom(t),
        lambda b: _get(b, STAT_INDICES["dynamic_range_bottom"]).min(),
    ),
    # Derived stats (no compute_fn, only combinator for final value)
    "mean": (
        None,
        lambda b: _get(b, STAT_INDICES["sum"]).sum() / _get(b, STAT_INDICES["numel"]).sum(),
    ),
    "variance": (
        None,
        lambda b: _combine_variance(b),
    ),
    "std": (
        None,
        lambda b: math.sqrt(max(0, _combine_variance(b))),
    ),
    "l2_norm": (
        None,
        lambda b: math.sqrt(float(_get(b, STAT_INDICES["l2_norm_sq"]).sum())),
    ),
    "dynamic_range": (
        None,
        lambda b: _get(b, STAT_INDICES["dynamic_range_top"]).max() -
                  _get(b, STAT_INDICES["dynamic_range_bottom"]).min(),
    ),
}


@torch.compile
def _compute_dr_top(t: torch.Tensor) -> torch.Tensor:
    abs_t = t.float().abs()
    nonzero = abs_t[abs_t > 0]
    if nonzero.numel() > 0:
        return torch.log2(nonzero.max())
    return torch.tensor(0.0, device=t.device)


@torch.compile
def _compute_dr_bottom(t: torch.Tensor) -> torch.Tensor:
    abs_t = t.float().abs()
    nonzero = abs_t[abs_t > 0]
    if nonzero.numel() > 0:
        return torch.log2(nonzero.min())
    return torch.tensor(0.0, device=t.device)


@torch.compile
def _combine_variance(b: torch.Tensor) -> float:
    total_numel = _get(b, STAT_INDICES["numel"]).sum()
    total_sum = _get(b, STAT_INDICES["sum"]).sum()
    total_sum_sq = _get(b, STAT_INDICES["sum_sq"]).sum()
    mean = total_sum / total_numel
    return ((total_sum_sq / total_numel) - mean ** 2).item()


DIRECT_STATS = {
    "entropy": lambda t: _compute_entropy(t),
    "kurtosis": lambda t: _compute_kurtosis(t),
    "median": lambda t: t.float().flatten().median().item(),
    "max_median_ratio": lambda t: _compute_max_median_ratio(t),
}


@torch.no_grad()
def _compute_entropy(t: torch.Tensor) -> float:
    p = t.float().flatten().clamp(min=1e-10)
    if p.sum() > 0:
        p = p / p.sum()
    return -(p * torch.log(p)).sum().item()


@torch.no_grad()
def _compute_kurtosis(t: torch.Tensor) -> float:
    flat = t.float().flatten()
    std = flat.std()
    if std > 1e-10:
        return ((flat - flat.mean()) ** 4).mean().item() / (std.item() ** 4)
    return 0.0


@torch.no_grad()
def _compute_max_median_ratio(t: torch.Tensor) -> float:
    flat = t.float().flatten()
    median = flat.median()
    if median != 0:
        return (flat.max() / median).item()
    return float('inf')
