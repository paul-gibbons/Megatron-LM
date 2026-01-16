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

"""LogMCoreTensorStats feature for logging MCore tensor statistics."""

import logging
from typing import Dict, Optional, Tuple

import torch

import nvdlfw_inspect.api as debug_api
from nvdlfw_inspect.registry import Registry, api_method

from megatron.core.debug.features.api import MCoreConfigAPIMapper
from megatron.core.debug.features.utils.stats_buffer import MCORE_STATS_BUFFERS
from megatron.core.debug.utils import compute_next_enabled_iter


@Registry.register_feature(namespace="megatron_core")
class LogMCoreTensorStats(MCoreConfigAPIMapper):
    """Log tensor statistics for Megatron-Core modules.

    Collects stats for MoE router, embeddings, output layer, and backward tensors.
    Statistics are buffered across micro-batches and logged on debug_api.step().

    Supported stats:
        Basic: min, max, mean, std, sum, numel, variance
        Norms: l1_norm, l2_norm, cur_amax, dynamic_range
        Zero counting: num_zeros, num_zeros%, num_zeros[threshold]%
        Per-element: per_element, per_element%
        Vocab gradient: vocab_topk_l2_pct[k] (% of grad norm from top-k tokens)
        Local only (not reducible): median, max_median_ratio, entropy, kurtosis
    """

    _BACKWARD_TENSOR_NAMES = {"wgrad", "dgrad", "gradient"}
    _NON_REDUCIBLE_STATS = {"median", "max_median_ratio", "entropy", "kurtosis"}
    _warned_non_reducible = False

    def __init__(self):
        super().__init__()

    def _get_supported_stats_list(self) -> set:
        return {
            "min", "max", "mean", "std", "sum", "numel", "variance",
            "median", "max_median_ratio", "kurtosis",
            "l1_norm", "l2_norm", "cur_amax", "dynamic_range",
            "entropy", "per_element", "per_element%",
            "num_zeros", "num_zeros%",
        }

    def _check_log_frequency(self, config: Dict, iteration: int) -> Tuple[bool, Optional[int]]:
        freq = config.get("freq", 1)
        start_step = config.get("start_step", 0)
        end_step = config.get("end_step", -1)
        start_end_list = config.get("start_end_list", None)

        return compute_next_enabled_iter(
            start_step, end_step, start_end_list, freq, iteration
        )

    @api_method
    def inspect_tensor_enabled(
        self,
        config: Dict,
        layer_name: str,
        tensor_name: str,
        iteration: int,
        **kwargs,
    ) -> Tuple[bool, Optional[int]]:
        """Check if tensor inspection is enabled for this iteration."""
        should_run, next_iter = self._check_log_frequency(config, iteration)

        # Track next_iter for buffer reduction decisions
        MCORE_STATS_BUFFERS.layers_to_next_iter[layer_name] = next_iter

        debug_api.log_message(
            f"Feature={self.__class__.__name__}, API=inspect_tensor_enabled: "
            f"layer={layer_name}, tensor={tensor_name}, enabled={should_run}, "
            f"next_iter={next_iter}",
            layer_name=layer_name,
            level=logging.DEBUG,
        )

        return should_run, next_iter

    def _validate_stats(self, stats: list) -> None:
        """Validate that all requested stats are supported."""
        from megatron.core.debug.features.utils.stats_computation import (
            parse_num_zeros_stat, parse_vocab_topk_stat,
        )

        supported = self._get_supported_stats_list()
        for stat in stats:
            if parse_num_zeros_stat(stat) is not None:
                continue
            if parse_vocab_topk_stat(stat) is not None:
                continue
            if stat.lower() not in supported:
                raise ValueError(
                    f"[MCore Debug] Unsupported stat: '{stat}'. "
                    f"Supported stats: {sorted(supported)} "
                    f"(also supports num_zeros[threshold]%, vocab_topk_l2_pct[k])"
                )

    def _warn_non_reducible_stats(self, stats: list) -> None:
        """Warn once if non-reducible stats are used in distributed context."""
        if LogMCoreTensorStats._warned_non_reducible:
            return

        non_reducible_used = [
            s for s in stats if s.lower() in self._NON_REDUCIBLE_STATS
        ]
        if non_reducible_used:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if rank == 0:
                debug_api.log_message(
                    f"[MCore Debug] Stats {non_reducible_used} are local-only (not reduced across ranks).",
                    layer_name="",
                    level=logging.WARNING,
                )
            LogMCoreTensorStats._warned_non_reducible = True

    @api_method
    def inspect_tensor(
        self,
        config: Dict,
        layer_name: str,
        tensor_name: str,
        tensor: torch.Tensor,
        iteration: int,
        **kwargs,
    ) -> None:
        """Feed tensor statistics to buffer for later logging."""
        should_run, _ = self._check_log_frequency(config, iteration)
        if not should_run:
            return

        stats_to_collect = config.get("stats", ["min", "max", "mean"])
        self._validate_stats(stats_to_collect)
        self._warn_non_reducible_stats(stats_to_collect)

        debug_api.log_message(
            f"Feature={self.__class__.__name__}, API=inspect_tensor: "
            f"layer={layer_name}, tensor={tensor_name}, shape={tensor.shape}",
            layer_name=layer_name,
            level=logging.INFO,
        )
        reduction_group = kwargs.get("reduction_group", None)
        tp_group = kwargs.get("tp_group", None)
        skip_reduction = kwargs.get("skip_reduction", False)

        tensor_lower = tensor_name.lower()
        reduce_within_microbatch = tensor_lower not in ("weight",)
        start_step = config.get("start_step", None)
        end_step = config.get("end_step", None)
        start_end_list = config.get("start_end_list", None)
        if start_end_list is not None:
            start_end_list = tuple(tuple(int(x) for x in interval) for interval in start_end_list)
        options = (start_step, end_step, start_end_list)

        MCORE_STATS_BUFFERS.try_add_buffer(
            layer_name=layer_name,
            tensor_name=tensor_name,
            stats=stats_to_collect,
            options=options,
            reduction_group=reduction_group,
            reduce_within_microbatch=reduce_within_microbatch,
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
