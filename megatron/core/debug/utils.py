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

"""Utility functions for MCore debug/inspection framework."""

from typing import Dict, Optional, Tuple

import torch

from megatron.core.debug.debug_state import MCoreDebugState


def get_reduction_params(
    tensor_name: str,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Tuple[bool, Optional[torch.distributed.ProcessGroup], bool]:
    """Get statistics reduction parameters for a tensor."""
    import nvdlfw_inspect.api as debug_api

    skip_reduction = False
    reduction_group = debug_api.get_tensor_reduction_group()
    reduce_within_microbatch = tensor_name.lower() != "weight"

    if tensor_name.lower() == "weight":
        if MCoreDebugState.weight_tensor_tp_group_reduce:
            reduction_group = tp_group
        else:
            skip_reduction = True

    return skip_reduction, reduction_group, reduce_within_microbatch


class TensorInspectMixin:
    """Mixin class for MCore modules that support tensor inspection."""

    _debug_name: Optional[str] = None
    _next_debug_iter: Optional[int] = 0
    _debug_last_iteration: Optional[int] = None
    _debug_enabled_this_iter: bool = False
    _backward_hook_handles: Dict[str, torch.utils.hooks.RemovableHandle] = None

    def _get_debug_name(self) -> str:
        """Return the layer name for debug logging."""
        raise NotImplementedError

    def _get_reduction_group(self) -> Optional[torch.distributed.ProcessGroup]:
        """Return the TP process group for weight tensor reduction."""
        return None

    def _get_gradient_targets(self) -> Dict[str, torch.nn.Module]:
        """Return {tensor_name: module} for gradient hooks (wgrad/dgrad)."""
        return {}

    def _is_debug_iter(self) -> bool:
        MCoreDebugState.ensure_initialized()
        if not MCoreDebugState.debug_enabled:
            return False
        current_iter = MCoreDebugState.get_iteration()
        if self._debug_last_iteration != current_iter:
            self._debug_enabled_this_iter = (
                self._next_debug_iter is not None
                and current_iter >= self._next_debug_iter
            )
            self._debug_last_iteration = current_iter
        return self._debug_enabled_this_iter

    def _inspect_tensor(self, tensor_name: str, tensor: torch.Tensor) -> None:
        """Inspect a tensor and collect statistics."""
        if not self._is_debug_iter():
            return
        import nvdlfw_inspect.api as debug_api

        layer_name = self._get_debug_name()
        iteration = MCoreDebugState.get_iteration()

        result = debug_api.megatron_core.inspect_tensor_enabled(
            layer_name=layer_name, tensor_name=tensor_name, iteration=iteration
        )

        if isinstance(result, tuple):
            enabled, next_iter = result
            if next_iter is not None:
                if self._next_debug_iter is None:
                    self._next_debug_iter = next_iter
                else:
                    self._next_debug_iter = min(self._next_debug_iter, next_iter)
        else:
            enabled = result

        if enabled:
            skip_reduction, reduction_group, _ = get_reduction_params(
                tensor_name, tp_group=self._get_reduction_group()
            )
            debug_api.megatron_core.inspect_tensor(
                layer_name=layer_name,
                tensor_name=tensor_name,
                tensor=tensor,
                iteration=iteration,
                reduction_group=reduction_group,
                skip_reduction=skip_reduction,
            )

    def _inspect_backward_tensor(self, tensor_name: str, grad: torch.Tensor) -> None:
        """Inspect a backward tensor (wgrad/dgrad) and collect statistics."""
        if not self._is_debug_iter():
            return
        import nvdlfw_inspect.api as debug_api

        layer_name = self._get_debug_name()
        iteration = MCoreDebugState.get_iteration()

        result = debug_api.megatron_core.inspect_tensor_enabled(
            layer_name=layer_name, tensor_name=tensor_name, iteration=iteration
        )

        if isinstance(result, tuple):
            enabled, next_iter = result
            if next_iter is not None:
                if self._next_debug_iter is None:
                    self._next_debug_iter = next_iter
                else:
                    self._next_debug_iter = min(self._next_debug_iter, next_iter)
        else:
            enabled = result

        if enabled:
            reduction_group = debug_api.get_tensor_reduction_group()
            debug_api.megatron_core.inspect_tensor(
                layer_name=layer_name,
                tensor_name=tensor_name,
                tensor=grad,
                iteration=iteration,
                reduction_group=reduction_group,
                skip_reduction=reduction_group is None,
                tp_group=self._get_reduction_group(),
            )

    def _setup_backward_hooks(self) -> None:
        """Register backward hooks for gradient inspection."""
        MCoreDebugState.ensure_initialized()
        if not MCoreDebugState.debug_enabled:
            return

        if self._backward_hook_handles is None:
            self._backward_hook_handles = {}

        targets = self._get_gradient_targets()
        for tensor_name, module in targets.items():
            if module is None:
                continue

            if tensor_name.lower() == "wgrad":
                if not hasattr(module, 'weight') or module.weight is None:
                    continue
                def make_wgrad_hook(tname: str):
                    def hook(grad: torch.Tensor) -> torch.Tensor:
                        self._inspect_backward_tensor(tname, grad)
                        return grad
                    return hook
                handle = module.weight.register_hook(make_wgrad_hook(tensor_name))
            else:
                def make_dgrad_hook(tname: str):
                    def hook(_, grad_input, grad_output):
                        if grad_input and grad_input[0] is not None:
                            self._inspect_backward_tensor(tname, grad_input[0])
                    return hook
                handle = module.register_full_backward_hook(make_dgrad_hook(tensor_name))

            self._backward_hook_handles[tensor_name] = handle

    def _remove_backward_hooks(self) -> None:
        """Remove all registered backward hooks."""
        if self._backward_hook_handles is not None:
            for handle in self._backward_hook_handles.values():
                handle.remove()
            self._backward_hook_handles.clear()


class OptimizerInspectMixin:
    """Mixin class for optimizers that support parameter inspection."""

    _optim_next_debug_iter: Optional[int] = 0
    _optim_debug_last_iteration: Optional[int] = None
    _optim_debug_enabled_this_iter: bool = False

    def _is_optim_debug_iter(self) -> bool:
        """Check if this iteration should run optimizer debug inspection."""
        MCoreDebugState.ensure_initialized()
        if not MCoreDebugState.debug_enabled:
            return False
        current_iter = MCoreDebugState.get_iteration()
        if self._optim_debug_last_iteration != current_iter:
            self._optim_debug_enabled_this_iter = (
                self._optim_next_debug_iter is not None
                and current_iter >= self._optim_next_debug_iter
            )
            self._optim_debug_last_iteration = current_iter
        return self._optim_debug_enabled_this_iter

    def _inspect_optimizer_param(
        self,
        param_name: str,
        param: torch.Tensor,
        grad: Optional[torch.Tensor],
        optimizer_state: Dict,
        iteration: int,
        reduction_group: Optional[torch.distributed.ProcessGroup] = None,
        is_distributed_optimizer: Optional[bool] = None,
    ) -> None:
        """Inspect a parameter and collect optimizer statistics."""
        import nvdlfw_inspect.api as debug_api

        result = debug_api.megatron_core.inspect_optimizer_param_enabled(
            layer_name=param_name,
            param_name=param_name,
            iteration=iteration,
        )

        if isinstance(result, tuple):
            enabled, next_iter = result
            if next_iter is not None:
                if self._optim_next_debug_iter is None:
                    self._optim_next_debug_iter = next_iter
                else:
                    self._optim_next_debug_iter = min(self._optim_next_debug_iter, next_iter)
        else:
            enabled = result

        if enabled:
            if is_distributed_optimizer is None:
                config = getattr(self, "config", None)
                is_distributed_optimizer = bool(
                    getattr(config, "use_distributed_optimizer", False)
                )
            else:
                is_distributed_optimizer = bool(is_distributed_optimizer)
            debug_api.megatron_core.inspect_optimizer_param(
                layer_name=param_name,
                param_name=param_name,
                param=param,
                grad=grad,
                optimizer_state=optimizer_state,
                iteration=iteration,
                reduction_group=reduction_group,
                is_distributed_optimizer=is_distributed_optimizer,
            )


def compute_next_enabled_iter(
    start_step: Optional[int],
    end_step: Optional[int],
    start_end_list: Optional[list],
    freq: int,
    current_iter: int,
) -> Tuple[bool, Optional[int]]:
    start = start_step if start_step is not None else 0
    end = end_step if end_step is not None and end_step >= 0 else float("inf")

    if start_end_list is not None:
        for range_start, range_end in start_end_list:
            if range_start <= current_iter <= range_end:
                if current_iter % freq == 0:
                    next_iter = current_iter + freq
                    if next_iter > range_end:
                        for ns, _ in start_end_list:
                            if ns > current_iter:
                                return True, ns
                        return True, None
                    return True, next_iter
                else:
                    next_freq = ((current_iter // freq) + 1) * freq
                    if next_freq <= range_end:
                        return False, next_freq
                    for ns, ne in start_end_list:
                        if ns > current_iter:
                            return False, ns
                    return False, None

        for ns, ne in start_end_list:
            if ns > current_iter:
                return False, ns
        return False, None

    if current_iter < start:
        return False, start
    if current_iter > end:
        return False, None

    if current_iter % freq == 0:
        next_iter = current_iter + freq
        if next_iter > end:
            return True, None
        return True, next_iter
    else:
        next_freq = ((current_iter // freq) + 1) * freq
        if next_freq > end:
            return False, None
        return False, next_freq

