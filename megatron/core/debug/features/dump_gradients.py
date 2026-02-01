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
from megatron.core.debug.features.utils.grad_dump import (
    WGRAD_BUFFER,
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
        self._debug_logged_iteration = -1

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

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        debug_this_param = (self._debug_logged_iteration != iteration)
        if debug_this_param:
            self._debug_logged_iteration = iteration
            logger.info(
                f"[DumpWGrads DEBUG] rank={rank} iter={iteration} "
                f"ENTRY layer_name={layer_name} (local shard mode)"
            )

        grad = kwargs.get("grad")
        if grad is None:
            grad = getattr(param, "main_grad", None)
            if grad is None:
                grad = param.grad
        if grad is None:
            return

        save_dir = config.get("save_dir")

        WGRAD_BUFFER.add(
            layer_name=layer_name,
            grad=grad,
            iteration=iteration,
            save_dir=save_dir,
        )

    def flush_wgrads(self) -> None:
        if WGRAD_BUFFER.has_data:
            WGRAD_BUFFER.flush()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()


def flush_wgrad_buffer():
    """Flush the wgrad buffer."""
    feature = None
    namespace_data = Registry.data.get("megatron_core")
    if namespace_data is not None:
        feature = namespace_data.features.get("DumpWGrads")
    if feature is not None and hasattr(feature, "flush_wgrads"):
        feature.flush_wgrads()
        return

    if WGRAD_BUFFER.has_data:
        WGRAD_BUFFER.flush()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()


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
