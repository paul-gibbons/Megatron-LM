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

"""API definition for Megatron-Core nvdlfw_inspect integration."""

import copy
from typing import Dict, Optional, Tuple

import torch

from nvdlfw_inspect.base import BaseConfigAPIMapper, BaseNamespaceAPI
from nvdlfw_inspect.registry import Registry

from megatron.core.debug.debug_state import MCoreDebugState
from megatron.core.debug.features.utils.stats_buffer import MCORE_STATS_BUFFERS


class MCoreConfigAPIMapper(BaseConfigAPIMapper):
    """Config API mapper for Megatron-Core features."""

    def parse_config_and_api(self, config, **kwargs):
        """Process config and return True if config matches API args."""
        processed_config = None
        config_copy = copy.deepcopy(config)
        tensor_parsing = kwargs.get("tensor_parsing", False)

        if tensor_parsing:
            processed_config = self._process_tensor_config(
                config_copy, kwargs["tensor_name"]
            )

        if not processed_config:
            return False, None

        if "enabled" in processed_config:
            processed_config.pop("enabled")
        return True, processed_config


class MCoreDefaultFeatures:
    """Default API implementations."""

    def inspect_tensor_enabled(
        self,
        config: Dict,
        layer_name: str,
        tensor_name: str,
        iteration: int,
        **kwargs,
    ) -> Tuple[bool, Optional[int]]:
        return False, None

    def inspect_tensor(
        self,
        config: Dict,
        layer_name: str,
        tensor_name: str,
        tensor: torch.Tensor,
        iteration: int,
        **kwargs,
    ) -> None:
        pass


required_kwargs = {
    "inspect_tensor": ["tensor_name"],
    "inspect_tensor_enabled": ["tensor_name"],
}


@Registry.register_namespace_api(namespace="megatron_core")
class MegatronCoreAPI(BaseNamespaceAPI):
    """Megatron-Core namespace API for nvdlfw_inspect."""

    def __init__(self):
        BaseNamespaceAPI.__init__(self)
        self._default_api_impl = MCoreDefaultFeatures()
        self._cacheable_api_kwargs_map = {
            "inspect_tensor": ["tensor_name"],
            "inspect_tensor_enabled": ["tensor_name", "iteration"],
        }

    def is_multiple_feature_invocation_allowed(self, api_name):
        return api_name in {"inspect_tensor", "inspect_tensor_enabled"}

    def input_assertions_hook(self, api_name, **kwargs):
        if api_name in required_kwargs:
            for kwarg in required_kwargs[api_name]:
                assert kwarg in kwargs, (
                    f"[MCore Debug] Cannot route API. Provide {kwarg} in {api_name}."
                )

    def routing_condition(self, api_name, config, layer_name, feature_obj, **kwargs):
        tensor_parsing = "tensor_name" in required_kwargs.get(api_name, [])
        status, modified_config = feature_obj.parse_config_and_api(
            config, tensor_parsing=tensor_parsing, **kwargs
        )
        return status, modified_config

    def output_assertions_hook(self, api_name, ret, **kwargs):
        if "enabled" in api_name:
            assert isinstance(ret, (bool, tuple))
        if api_name == "inspect_tensor":
            assert ret is None

    def handle_multi_feature_output(
        self, api_name, multi_feature_outputs, features_to_invoke, **kwargs
    ):
        if "enabled" in api_name:
            all_ret_tuple = all(
                isinstance(output, tuple) for output in multi_feature_outputs
            )
            if all_ret_tuple:
                run_current = any(output[0] for output in multi_feature_outputs)
                next_iter = None
                for output in multi_feature_outputs:
                    if next_iter is None:
                        next_iter = output[1]
                    elif output[1] is not None:
                        next_iter = min(next_iter, output[1])
                return run_current, next_iter
            run_current = any(output for output in multi_feature_outputs)
            return run_current, None
        return super().handle_multi_feature_output(
            api_name, multi_feature_outputs, features_to_invoke, **kwargs
        )

    def step(self):
        current_iter = MCoreDebugState.get_iteration()
        MCORE_STATS_BUFFERS.log_stats(current_iter)

    def end_debug(self):
        MCORE_STATS_BUFFERS.reset()
        MCoreDebugState._reset()

