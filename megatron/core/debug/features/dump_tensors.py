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

"""DumpTensors feature for saving tensors to disk during training."""

import fnmatch
import logging
from typing import Dict, List, Optional, Tuple

import torch

from nvdlfw_inspect.registry import Registry, api_method

from megatron.core.debug.features.api import MCoreConfigAPIMapper
from megatron.core.debug.features.utils.tensor_dump import (
    TensorDumpBuffer,
    TENSOR_DUMP_STATE,
    save_tensor_dump,
    save_tensor_direct,
)

logger = logging.getLogger(__name__)


@Registry.register_feature(namespace="megatron_core")
class DumpTensors(MCoreConfigAPIMapper):
    """Dump tensors to disk via inspect_tensor API."""

    def __init__(self):
        super().__init__()
        self._warned_no_save_dir = False

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
        should_run, next_iter = self._check_log_frequency(config, iteration)
        if not should_run:
            return False, next_iter

        save_dir = config.get("save_dir")
        if not save_dir:
            if not self._warned_no_save_dir:
                logger.warning("[DumpTensors] save_dir not configured")
                self._warned_no_save_dir = True
            return False, next_iter

        if not self._matches_pattern(layer_name, config.get("layers", ["*"])):
            return False, next_iter

        if not self._matches_pattern(tensor_name, config.get("tensors", ["*"])):
            return False, next_iter

        TENSOR_DUMP_STATE.save_dir = save_dir
        TENSOR_DUMP_STATE.current_iteration = iteration
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
        enabled, _ = self.inspect_tensor_enabled(
            config, layer_name, tensor_name, iteration, **kwargs
        )
        if not enabled:
            return

        save_tensor_direct(
            config.get("save_dir"), iteration, layer_name, tensor_name, tensor,
            log_to_metrics=False,
        )


def flush_tensor_dumps(iteration: Optional[int] = None) -> None:
    """Flush tensor dumps to disk at end of iteration."""
    from megatron.core.debug.debug_state import MCoreDebugState

    MCoreDebugState.ensure_initialized()
    if not MCoreDebugState.debug_enabled:
        return

    if iteration is None:
        iteration = MCoreDebugState.get_iteration()

    if TENSOR_DUMP_STATE.has_data() and TENSOR_DUMP_STATE.save_dir:
        save_tensor_dump(TENSOR_DUMP_STATE.save_dir, iteration)
