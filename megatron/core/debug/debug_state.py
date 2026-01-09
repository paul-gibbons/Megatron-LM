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

"""MCoreDebugState - manages debug state for Megatron-Core modules."""

import sys
from typing import Optional


class MCoreDebugState:
    """Manages debug state for Megatron-Core modules."""
    debug_enabled: Optional[bool] = None
    layer_count: int = 1
    layers_initialized: dict = {}
    weight_tensor_tp_group_reduce: bool = True

    @classmethod
    def initialize(cls) -> None:
        if "nvdlfw_inspect" in sys.modules:
            import nvdlfw_inspect.api as debug_api

            if cls.debug_enabled is False and debug_api.DEBUG_MANAGER is not None:
                raise RuntimeError(
                    "[MCore Debug] nvdlfw_inspect should be initialized before "
                    "initialization of the first MCore module with debug support"
                )
            cls.debug_enabled = debug_api.DEBUG_MANAGER is not None
        else:
            cls.debug_enabled = False

    @classmethod
    def _reset(cls) -> None:
        cls.debug_enabled = None
        cls.layer_count = 1
        cls.layers_initialized.clear()
        cls.weight_tensor_tp_group_reduce = True

    @classmethod
    def set_weight_tensor_tp_group_reduce(cls, enabled: bool) -> None:
        """Set whether weight tensor stats should be reduced across TP group."""
        cls.weight_tensor_tp_group_reduce = enabled

    @classmethod
    def get_layer_count(cls) -> int:
        count = cls.layer_count
        cls.layer_count += 1
        return count

    @classmethod
    def get_iteration(cls) -> int:
        if not cls.debug_enabled:
            return 0
        import nvdlfw_inspect.api as debug_api

        return debug_api.DEBUG_MANAGER._trainer_iteration_count

    @classmethod
    def is_initialized(cls) -> bool:
        return cls.debug_enabled is not None

    @classmethod
    def ensure_initialized(cls) -> None:
        if not cls.is_initialized():
            cls.initialize()

