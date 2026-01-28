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

"""LogVocabGradientStats - Simplified per-vocabulary gradient analysis."""

import copy
from typing import Dict, Optional, Tuple

import torch

from nvdlfw_inspect.registry import Registry, api_method

from megatron.core.debug.features.api import MCoreConfigAPIMapper
from megatron.core.debug.features.utils.stats_buffer import MCORE_STATS_BUFFERS
from megatron.core.debug.utils import compute_next_enabled_iter


@Registry.register_feature(namespace="megatron_core")
class LogVocabGradientStats(MCoreConfigAPIMapper):
    """Per-vocabulary gradient analysis for output layer wgrad.

    Analyzes which tokens contribute most to gradient norm - useful for
    detecting training instability where few tokens dominate the gradient.

    Config:
        topk: List of k values for cumulative L2% (default: [1, 5, 10, 50, 100])
        freq: Logging frequency (default: 1)
        start_step/end_step: Step range (default: 0, -1)

    Output: vocab_topk_l2_pct[k] - cumulative % of gradient L2 from top-k tokens
    """

    def parse_config_and_api(self, config, **kwargs):
        """Override to bypass tensor config validation - this feature targets wgrad only."""
        tensor_name = kwargs.get("tensor_name", "")
        if tensor_name.lower() != "wgrad":
            return False, None
        config_copy = copy.deepcopy(config)
        if "enabled" in config_copy:
            config_copy.pop("enabled")
        return True, config_copy

    def _check_log_frequency(self, config: Dict, iteration: int) -> Tuple[bool, Optional[int]]:
        return compute_next_enabled_iter(
            config.get("start_step", 0),
            config.get("end_step", -1),
            config.get("start_end_list"),
            config.get("freq", 1),
            iteration,
        )

    @api_method
    def inspect_tensor_enabled(
        self, config: Dict, layer_name: str, tensor_name: str, iteration: int, **kwargs
    ) -> Tuple[bool, Optional[int]]:
        if tensor_name.lower() != "wgrad":
            return False, None
        should_run, next_iter = self._check_log_frequency(config, iteration)
        MCORE_STATS_BUFFERS.layers_to_next_iter[layer_name] = next_iter
        return should_run, next_iter

    @api_method
    def inspect_tensor(
        self, config: Dict, layer_name: str, tensor_name: str, tensor: torch.Tensor,
        iteration: int, **kwargs
    ) -> None:
        if tensor_name.lower() != "wgrad" or tensor.dim() != 2:
            return

        should_run, _ = self._check_log_frequency(config, iteration)
        if not should_run:
            return

        topk = [int(k) for k in config.get("topk", [1, 5, 10, 50, 100])]
        topk = sorted({k for k in topk if k > 0})
        stats = [f"vocab_topk_l2_pct[{k}]" for k in topk]

        start_end_list = config.get("start_end_list", None)
        if start_end_list is not None:
            start_end_list = tuple(tuple(int(x) for x in interval) for interval in start_end_list)
        options = (config.get("start_step", 0), config.get("end_step", -1), start_end_list)

        tp_group = kwargs.get("tp_group")
        reduction_group = kwargs.get("reduction_group")
        skip_reduction = tp_group is None and reduction_group is None

        MCORE_STATS_BUFFERS.try_add_buffer(
            layer_name=layer_name,
            tensor_name=tensor_name,
            stats=stats,
            options=options,
            reduction_group=reduction_group,
            reduce_within_microbatch=True,
            tp_group=tp_group,
        )

        MCORE_STATS_BUFFERS.feed(
            layer_name=layer_name,
            tensor_name=tensor_name,
            options=options,
            tensor=tensor,
            iteration=iteration,
            skip_reduction=skip_reduction,
        )
