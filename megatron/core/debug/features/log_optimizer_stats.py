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

"""LogOptimizerStats feature for optimizer statistics."""

from typing import Dict, Optional, Tuple

import torch

from nvdlfw_inspect.registry import Registry, api_method

from megatron.core.debug.features.api import MCoreConfigAPIMapper
from megatron.core.debug.features.utils.optimizer_stats_buffer import OPTIMIZER_STATS_BUFFERS
from megatron.core.debug.utils import compute_next_enabled_iter


@Registry.register_feature(namespace="megatron_core")
class LogOptimizerStats(MCoreConfigAPIMapper):
    """Log optimizer statistics per parameter or aggregated by layer.

    Supported stats:
        num_zeros, num_zeros%, num_zeros[threshold]%, grad_norm, weight_grad_ratio,
        exp_avg_norm, exp_avg_sq_mean, rms_staleness, update_norm
    """

    _SUPPORTED_STATS = {
        "num_zeros", "num_zeros%", "grad_norm", "weight_grad_ratio",
        "exp_avg_norm", "exp_avg_sq_mean", "grad_to_v_ratio", "rms_staleness", "update_norm",
    }

    def _check_log_frequency(self, config: Dict, iteration: int) -> Tuple[bool, Optional[int]]:
        return compute_next_enabled_iter(
            config.get("start_step", 0),
            config.get("end_step", -1),
            config.get("start_end_list"),
            config.get("freq", 1),
            iteration,
        )

    def _validate_stats(self, stats: list) -> None:
        from megatron.core.debug.features.utils.stats_buffer import parse_num_zeros_stat
        for stat in stats:
            if parse_num_zeros_stat(stat) is None and stat.lower() not in self._SUPPORTED_STATS:
                raise ValueError(f"Unsupported optimizer stat: '{stat}'")

    @api_method
    def inspect_optimizer_param_enabled(
        self, config: Dict, layer_name: str, iteration: int, **kwargs
    ) -> Tuple[bool, Optional[int]]:
        return self._check_log_frequency(config, iteration)

    @api_method
    def inspect_optimizer_param(
        self, config: Dict, layer_name: str, param: torch.Tensor, iteration: int, **kwargs
    ) -> None:
        should_run, _ = self._check_log_frequency(config, iteration)
        if not should_run:
            return

        stats = config.get("stats", ["num_zeros%", "grad_norm"])
        self._validate_stats(stats)

        OPTIMIZER_STATS_BUFFERS.feed(
            param_name=kwargs.get("param_name", layer_name),
            param=param,
            grad=kwargs.get("grad"),
            optimizer_state=kwargs.get("optimizer_state", {}),
            stats=stats,
            iteration=iteration,
            reduction_group=kwargs.get("reduction_group"),
            aggregate_by=config.get("aggregate_by"),
        )
