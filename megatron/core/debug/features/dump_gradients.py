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

"""Gradient dumping features for saving wgrads and dgrads during training."""

import fnmatch
import logging
from typing import Dict, List, Optional, Tuple

import torch

from nvdlfw_inspect.registry import Registry, api_method

from megatron.core.debug.features.api import MCoreConfigAPIMapper
from megatron.core.debug.features.utils.tensor_dump import save_tensor_direct
from megatron.core.debug.features.utils.dgrad_logger import (
    DGRAD_LOGGER,
    DGradLogger,
    register_dgrad_hooks,
    remove_dgrad_hooks,
    save_dgrads,
    enable_dgrad_capture,
    disable_dgrad_capture,
)

logger = logging.getLogger(__name__)


@Registry.register_feature(namespace="megatron_core")
class DumpWGrads(MCoreConfigAPIMapper):
    """Dump weight gradients to disk via inspect_optimizer_param API."""

    def __init__(self):
        super().__init__()
        self._warned_no_save_dir = False
        self._debug_logged_iteration = -1  # Track which iteration we've logged debug info for

    def parse_config_and_api(self, config, **kwargs):
        if kwargs.get("tensor_parsing", False):
            return False, None
        return super().parse_config_and_api(config, **kwargs)

    def _matches_pattern(self, name: str, patterns: List[str]) -> bool:
        if not patterns or "*" in patterns:
            return True
        return any(fnmatch.fnmatch(name, p) for p in patterns)

    @api_method
    def inspect_optimizer_param_enabled(
        self,
        config: Dict,
        layer_name: str,
        iteration: int,
        **_kwargs,
    ) -> Tuple[bool, Optional[int]]:
        should_run, next_iter = self._check_log_frequency(config, iteration)

        if not should_run:
            return False, next_iter

        save_dir = config.get("save_dir")
        if not save_dir:
            if not self._warned_no_save_dir:
                logger.warning("[DumpWGrads] save_dir not configured, wgrad dumping disabled")
                self._warned_no_save_dir = True
            return False, next_iter

        if not self._matches_pattern(layer_name, config.get("layers", ["*"])):
            return False, next_iter

        return True, next_iter

    @api_method
    def inspect_optimizer_param(
        self,
        config: Dict,
        layer_name: str,
        param: torch.Tensor,
        iteration: int,
        **kwargs,
    ) -> None:
        enabled, _ = self.inspect_optimizer_param_enabled(
            config, layer_name, iteration, **kwargs
        )
        if not enabled:
            return

        # Debug logging - log once per iteration for first param to avoid spam
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        debug_this_param = (self._debug_logged_iteration != iteration)
        if debug_this_param:
            self._debug_logged_iteration = iteration
            logger.info(
                f"[DumpWGrads DEBUG] rank={rank} iter={iteration} "
                f"ENTRY layer_name={layer_name} kwargs_keys={list(kwargs.keys())}"
            )

        grad = getattr(param, "main_grad", None)
        if grad is None:
            grad = param.grad
        if grad is None:
            if debug_this_param:
                logger.warning(
                    f"[DumpWGrads DEBUG] rank={rank} iter={iteration} "
                    f"SKIP layer={layer_name} - grad is None! "
                    f"param.main_grad={getattr(param, 'main_grad', 'NO_ATTR')} "
                    f"param.grad={param.grad}"
                )
            return

        save_dir = config.get("save_dir")
        is_distributed_optimizer = kwargs.get("is_distributed_optimizer", False)
        reduction_group = kwargs.get("reduction_group")
        is_expert_parallel = kwargs.get("is_expert_parallel", False)
        skip_expert_all_gather = config.get("skip_expert_all_gather", False)
        effective_group = reduction_group

        if debug_this_param:
            # Get parallel state info for debugging
            try:
                from megatron.core import parallel_state as mpu
                dp_rank = mpu.get_data_parallel_rank()
                dp_world = mpu.get_data_parallel_world_size()
                tp_rank = mpu.get_tensor_model_parallel_rank()
                pp_rank = mpu.get_pipeline_model_parallel_rank()
            except Exception as e:
                dp_rank = dp_world = tp_rank = pp_rank = -1
                logger.warning(f"[DumpWGrads DEBUG] Could not get parallel state: {e}")

            reduction_group_info = "None"
            if reduction_group is not None:
                try:
                    reduction_group_info = f"size={torch.distributed.get_world_size(reduction_group)}"
                except Exception:
                    reduction_group_info = "error_getting_size"

            logger.info(
                f"[DumpWGrads DEBUG] rank={rank} iter={iteration} layer={layer_name}\n"
                f"  is_distributed_optimizer={is_distributed_optimizer}\n"
                f"  reduction_group={reduction_group_info}\n"
                f"  is_expert_parallel={is_expert_parallel}\n"
                f"  skip_expert_all_gather={skip_expert_all_gather}\n"
                f"  grad.shape={grad.shape} grad.dtype={grad.dtype}\n"
                f"  dp_rank={dp_rank} dp_world={dp_world} tp_rank={tp_rank} pp_rank={pp_rank}"
            )

        if is_expert_parallel:
            try:
                from megatron.core import parallel_state as mpu
                effective_group = mpu.get_expert_data_parallel_group()
            except (AssertionError, RuntimeError):
                effective_group = reduction_group

        if is_expert_parallel and skip_expert_all_gather:
            # Save local shard to avoid large all_gather for expert-parallel weights.
            full_grad = grad
            if debug_this_param:
                logger.info(
                    f"[DumpWGrads DEBUG] rank={rank} iter={iteration} "
                    f"PATH: skip_expert_all_gather - saving local shard"
                )
        elif is_distributed_optimizer and effective_group is not None:
            world_size = torch.distributed.get_world_size(effective_group)
            if debug_this_param:
                logger.info(
                    f"[DumpWGrads DEBUG] rank={rank} iter={iteration} "
                    f"PATH: distributed_optimizer all_gather - world_size={world_size}"
                )
            if world_size > 1:
                if debug_this_param:
                    logger.info(
                        f"[DumpWGrads DEBUG] rank={rank} iter={iteration} "
                        f"BEFORE all_gather: grad.shape={grad.shape}"
                    )
                gathered_grads = [torch.empty_like(grad) for _ in range(world_size)]
                torch.distributed.all_gather(gathered_grads, grad, group=effective_group)
                full_grad = torch.cat(gathered_grads, dim=0)
                if debug_this_param:
                    logger.info(
                        f"[DumpWGrads DEBUG] rank={rank} iter={iteration} "
                        f"AFTER all_gather: full_grad.shape={full_grad.shape}"
                    )
            else:
                full_grad = grad
        else:
            full_grad = grad
            if debug_this_param:
                logger.info(
                    f"[DumpWGrads DEBUG] rank={rank} iter={iteration} "
                    f"PATH: no all_gather needed - using grad directly"
                )

        if debug_this_param:
            logger.info(
                f"[DumpWGrads DEBUG] rank={rank} iter={iteration} "
                f"SAVING to {save_dir}/wgrads layer={layer_name} shape={full_grad.shape}"
            )

        save_tensor_direct(
            f"{save_dir}/wgrads", iteration, layer_name, "main_grad", full_grad,
            log_to_metrics=False,
            include_microbatch=False,
            track_state=False,
        )


@Registry.register_feature(namespace="megatron_core")
class DumpDGrads(MCoreConfigAPIMapper):
    """Dump activation gradients (dgrads) to disk via backward hooks."""

    def __init__(self):
        super().__init__()
        self._warned_no_save_dir = False

    def parse_config_and_api(self, config, **kwargs):
        if kwargs.get("param_parsing", False):
            return False, None

        if kwargs.get("tensor_parsing", False):
            import copy
            config_copy = copy.deepcopy(config)
            config_copy.pop("enabled", None)
            return True, config_copy

        return super().parse_config_and_api(config, **kwargs)

    def _matches_pattern(self, name: str, patterns: List[str]) -> bool:
        if not patterns or "*" in patterns:
            return True
        return any(fnmatch.fnmatch(name, p) for p in patterns)

    @api_method
    def inspect_tensor_enabled(
        self,
        config: Dict,
        layer_name: str,
        tensor_name: str,
        iteration: int,
        **kwargs,
    ) -> Tuple[bool, Optional[int]]:
        del layer_name, tensor_name, kwargs
        should_run, next_iter = self._check_log_frequency(config, iteration)

        if not should_run:
            return False, next_iter

        save_dir = config.get("save_dir")
        if not save_dir:
            if not self._warned_no_save_dir:
                logger.warning("[DumpDGrads] save_dir not configured")
                self._warned_no_save_dir = True
            return False, next_iter

        DGRAD_LOGGER.enable(save_dir, iteration)
        return True, next_iter

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
        del config, layer_name, tensor_name, tensor, iteration, kwargs


__all__ = [
    "DumpWGrads",
    "DumpDGrads",
    "DGradLogger",
    "DGRAD_LOGGER",
    "register_dgrad_hooks",
    "remove_dgrad_hooks",
    "save_dgrads",
    "enable_dgrad_capture",
    "disable_dgrad_capture",
]
