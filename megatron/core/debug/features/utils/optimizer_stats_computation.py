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

"""Stat computation functions for optimizer statistics."""

from typing import Dict, List, Optional

import torch

from megatron.core.debug.features.utils.stats_computation import parse_num_zeros_stat

# Buffer indices for optimizer stats (num_zeros handled separately via dict)
STAT_INDICES = {
    "numel": 0,
    "grad_norm_sq": 1,
    "param_norm_sq": 2,
    "sum_g2_over_v": 3,
    "exp_avg_norm_sq": 4,
    "exp_avg_sq_sum": 5,
    "update_norm_sq": 6,
}
NUM_BUFFER_STATS = len(STAT_INDICES)


def compute_buffer_stats(
    param: torch.Tensor,
    grad: Optional[torch.Tensor],
    optimizer_state: Dict,
    stats: List[str],
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute raw components and store in buffer tensor (num_zeros handled separately)."""
    buffer = torch.zeros(NUM_BUFFER_STATS, dtype=torch.float64, device="cuda")
    requested = set(s.lower() for s in stats)

    buffer[STAT_INDICES["numel"]] = float(param.numel())

    if grad is not None:
        grad_f = grad.float()

        if "grad_norm" in requested or "weight_grad_ratio" in requested:
            buffer[STAT_INDICES["grad_norm_sq"]] = (grad_f ** 2).sum().item()
        if "weight_grad_ratio" in requested:
            buffer[STAT_INDICES["param_norm_sq"]] = (param.float() ** 2).sum().item()

        exp_avg_sq = optimizer_state.get("exp_avg_sq")
        if exp_avg_sq is not None and requested & {"rms_staleness", "grad_to_v_ratio"}:
            buffer[STAT_INDICES["sum_g2_over_v"]] = ((grad_f ** 2) / (exp_avg_sq.float() + eps)).sum().item()

    exp_avg = optimizer_state.get("exp_avg")
    exp_avg_sq = optimizer_state.get("exp_avg_sq")

    if exp_avg is not None and "exp_avg_norm" in requested:
        buffer[STAT_INDICES["exp_avg_norm_sq"]] = (exp_avg.float() ** 2).sum().item()
    if exp_avg_sq is not None and "exp_avg_sq_mean" in requested:
        buffer[STAT_INDICES["exp_avg_sq_sum"]] = exp_avg_sq.float().sum().item()
    if exp_avg is not None and exp_avg_sq is not None and "update_norm" in requested:
        update = exp_avg.float() / (exp_avg_sq.float().sqrt() + eps)
        buffer[STAT_INDICES["update_norm_sq"]] = (update ** 2).sum().item()

    return buffer


def compute_final_stats(buffer: torch.Tensor, stats: List[str]) -> Dict[str, float]:
    """Compute final stats from buffer (num_zeros handled separately)."""
    results = {}
    numel = buffer[STAT_INDICES["numel"]].item()
    if numel == 0:
        numel = 1.0

    for stat in stats:
        s = stat.lower()
        if parse_num_zeros_stat(stat):
            continue
        elif s == "grad_norm":
            results["grad_norm"] = buffer[STAT_INDICES["grad_norm_sq"]].item() ** 0.5
        elif s == "weight_grad_ratio":
            g = buffer[STAT_INDICES["grad_norm_sq"]].item() ** 0.5
            results["weight_grad_ratio"] = buffer[STAT_INDICES["param_norm_sq"]].item() ** 0.5 / g if g > 0 else float('inf')
        elif s in ("rms_staleness", "grad_to_v_ratio"):
            results[stat] = (buffer[STAT_INDICES["sum_g2_over_v"]].item() / numel) ** 0.5
        elif s == "exp_avg_norm":
            results["exp_avg_norm"] = buffer[STAT_INDICES["exp_avg_norm_sq"]].item() ** 0.5
        elif s == "exp_avg_sq_mean":
            results["exp_avg_sq_mean"] = buffer[STAT_INDICES["exp_avg_sq_sum"]].item() / numel
        elif s == "update_norm":
            results["update_norm"] = buffer[STAT_INDICES["update_norm_sq"]].item() ** 0.5

    return results
