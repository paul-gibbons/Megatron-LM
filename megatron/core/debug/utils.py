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

"""Utility functions for MCore debug/inspection framework.

This module provides standalone functions for tensor and optimizer inspection
without requiring mixin inheritance in model classes.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

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


# ---------------------------------------------------------------------------
# Registry for per-layer and per-optimizer debug state
# ---------------------------------------------------------------------------


@dataclass
class LayerDebugState:
    """Debug state for a single layer."""

    next_debug_iter: Optional[int] = 0
    last_iteration: Optional[int] = None
    enabled_this_iter: bool = False
    backward_hook_handles: Dict[str, torch.utils.hooks.RemovableHandle] = field(
        default_factory=dict
    )

    def update_for_iteration(self, current_iter: int) -> bool:
        """Update state for new iteration and return whether debug is enabled."""
        if self.last_iteration != current_iter:
            self.enabled_this_iter = (
                self.next_debug_iter is not None and current_iter >= self.next_debug_iter
            )
            self.last_iteration = current_iter
        return self.enabled_this_iter


@dataclass
class OptimizerDebugState:
    """Debug state for a single optimizer."""

    next_debug_iter: Optional[int] = 0
    last_iteration: Optional[int] = None
    enabled_this_iter: bool = False

    def update_for_iteration(self, current_iter: int) -> bool:
        """Update state for new iteration and return whether debug is enabled."""
        if self.last_iteration != current_iter:
            self.enabled_this_iter = (
                self.next_debug_iter is not None and current_iter >= self.next_debug_iter
            )
            self.last_iteration = current_iter
        return self.enabled_this_iter


class TensorInspectRegistry:
    """Central registry for per-layer and per-optimizer debug state."""

    _layer_state: Dict[str, LayerDebugState] = {}
    _optim_state: Dict[int, OptimizerDebugState] = {}

    @classmethod
    def get_layer_state(cls, layer_name: str) -> LayerDebugState:
        """Get or create debug state for a layer."""
        if layer_name not in cls._layer_state:
            cls._layer_state[layer_name] = LayerDebugState()
        return cls._layer_state[layer_name]

    @classmethod
    def get_optim_state(cls, optimizer_id: int) -> OptimizerDebugState:
        """Get or create debug state for an optimizer."""
        if optimizer_id not in cls._optim_state:
            cls._optim_state[optimizer_id] = OptimizerDebugState()
        return cls._optim_state[optimizer_id]

    @classmethod
    def reset(cls) -> None:
        """Reset all debug state (for testing)."""
        # Remove any registered hooks before clearing
        for state in cls._layer_state.values():
            for handle in state.backward_hook_handles.values():
                handle.remove()
            state.backward_hook_handles.clear()
        cls._layer_state.clear()
        cls._optim_state.clear()


# ---------------------------------------------------------------------------
# Standalone functions for tensor inspection
# ---------------------------------------------------------------------------


def is_debug_iter(layer_name: str) -> bool:
    """Check if this iteration should run tensor debug inspection for the layer.

    Args:
        layer_name: The layer name to check.

    Returns:
        True if debug inspection should run this iteration.
    """
    MCoreDebugState.ensure_initialized()
    if not MCoreDebugState.debug_enabled:
        state = TensorInspectRegistry.get_layer_state(layer_name)
        if state.backward_hook_handles:
            remove_backward_hooks(layer_name)
        return False

    state = TensorInspectRegistry.get_layer_state(layer_name)
    return state.update_for_iteration(MCoreDebugState.get_iteration())


def inspect_tensor(
    layer_name: str,
    tensor_name: str,
    tensor: torch.Tensor,
    reduction_group: Optional[torch.distributed.ProcessGroup] = None,
) -> None:
    """Inspect a tensor and collect statistics.

    This is the main entry point for tensor inspection. It handles:
    - Iteration gating (only runs on enabled iterations)
    - Calling the debug API to check if inspection is enabled
    - Updating next_debug_iter based on API response
    - Actually collecting tensor stats if enabled

    Args:
        layer_name: The layer name for debug logging (e.g., "embedding", "decoder.layers.0.mlp.router").
        tensor_name: The tensor name (e.g., "word", "logits", "router_probs").
        tensor: The tensor to inspect.
        reduction_group: Optional process group for tensor parallel reduction.
    """
    if not is_debug_iter(layer_name):
        return

    import nvdlfw_inspect.api as debug_api

    iteration = MCoreDebugState.get_iteration()
    state = TensorInspectRegistry.get_layer_state(layer_name)

    result = debug_api.megatron_core.inspect_tensor_enabled(
        layer_name=layer_name, tensor_name=tensor_name, iteration=iteration
    )

    if isinstance(result, tuple):
        enabled, next_iter = result
        _update_next_debug_iter(state, next_iter, iteration)
    else:
        enabled = result

    if enabled:
        skip_reduction, effective_reduction_group, _ = get_reduction_params(
            tensor_name, tp_group=reduction_group
        )
        debug_api.megatron_core.inspect_tensor(
            layer_name=layer_name,
            tensor_name=tensor_name,
            tensor=tensor,
            iteration=iteration,
            reduction_group=effective_reduction_group,
            skip_reduction=skip_reduction,
        )


def inspect_backward_tensor(
    layer_name: str,
    tensor_name: str,
    grad: torch.Tensor,
    reduction_group: Optional[torch.distributed.ProcessGroup] = None,
) -> None:
    """Inspect a backward tensor (wgrad/dgrad) and collect statistics.

    Args:
        layer_name: The layer name for debug logging.
        tensor_name: The tensor name (e.g., "wgrad", "dgrad").
        grad: The gradient tensor to inspect.
        reduction_group: Optional process group for tensor parallel reduction.
    """
    if not is_debug_iter(layer_name):
        return

    import nvdlfw_inspect.api as debug_api

    iteration = MCoreDebugState.get_iteration()
    state = TensorInspectRegistry.get_layer_state(layer_name)

    result = debug_api.megatron_core.inspect_tensor_enabled(
        layer_name=layer_name, tensor_name=tensor_name, iteration=iteration
    )

    if isinstance(result, tuple):
        enabled, next_iter = result
        _update_next_debug_iter(state, next_iter, iteration)
    else:
        enabled = result

    if enabled:
        dp_reduction_group = debug_api.get_tensor_reduction_group()
        debug_api.megatron_core.inspect_tensor(
            layer_name=layer_name,
            tensor_name=tensor_name,
            tensor=grad,
            iteration=iteration,
            reduction_group=dp_reduction_group,
            skip_reduction=dp_reduction_group is None,
            tp_group=reduction_group,
        )


def setup_backward_hooks(
    layer_name: str,
    gradient_targets: Dict[str, torch.nn.Module],
    reduction_group: Optional[torch.distributed.ProcessGroup] = None,
) -> None:
    """Register backward hooks for gradient inspection.

    Args:
        layer_name: The layer name for debug logging.
        gradient_targets: Dict mapping tensor_name to module for gradient hooks.
            Example: {"wgrad": self.word_embeddings, "dgrad": self}
        reduction_group: Optional process group for tensor parallel reduction.
    """
    MCoreDebugState.ensure_initialized()
    if not MCoreDebugState.debug_enabled:
        return

    state = TensorInspectRegistry.get_layer_state(layer_name)

    for tensor_name, module in gradient_targets.items():
        if module is None:
            continue
        if tensor_name in state.backward_hook_handles:
            continue

        if tensor_name.lower() == "wgrad":
            if not hasattr(module, "weight") or module.weight is None:
                continue

            def make_wgrad_hook(lname: str, tname: str, rgroup):
                def hook(grad: torch.Tensor) -> torch.Tensor:
                    inspect_backward_tensor(lname, tname, grad, reduction_group=rgroup)
                    return grad

                return hook

            handle = module.weight.register_hook(
                make_wgrad_hook(layer_name, tensor_name, reduction_group)
            )
        else:

            def make_dgrad_hook(lname: str, tname: str, rgroup):
                def hook(_, grad_input, grad_output):
                    if grad_input and grad_input[0] is not None:
                        inspect_backward_tensor(
                            lname, tname, grad_input[0], reduction_group=rgroup
                        )

                return hook

            handle = module.register_full_backward_hook(
                make_dgrad_hook(layer_name, tensor_name, reduction_group)
            )

        state.backward_hook_handles[tensor_name] = handle


def remove_backward_hooks(layer_name: str) -> None:
    """Remove all registered backward hooks for a layer.

    Args:
        layer_name: The layer name whose hooks should be removed.
    """
    state = TensorInspectRegistry.get_layer_state(layer_name)
    for handle in state.backward_hook_handles.values():
        handle.remove()
    state.backward_hook_handles.clear()


def manage_backward_hooks(
    layer_name: str,
    gradient_targets: Dict[str, torch.nn.Module],
    reduction_group: Optional[torch.distributed.ProcessGroup] = None,
) -> None:
    """Dynamically manage backward hooks based on debug state.

    Call this at the start of forward() to set up or remove hooks as needed.

    Args:
        layer_name: The layer name for debug logging.
        gradient_targets: Dict mapping tensor_name to module for gradient hooks.
        reduction_group: Optional process group for tensor parallel reduction.
    """
    MCoreDebugState.ensure_initialized()
    if not MCoreDebugState.debug_enabled:
        remove_backward_hooks(layer_name)
        return

    state = TensorInspectRegistry.get_layer_state(layer_name)
    prev_iteration = state.last_iteration
    enabled = state.update_for_iteration(MCoreDebugState.get_iteration())

    if prev_iteration != state.last_iteration:
        if enabled and not state.backward_hook_handles:
            setup_backward_hooks(layer_name, gradient_targets, reduction_group)
        elif not enabled and state.backward_hook_handles:
            remove_backward_hooks(layer_name)


# ---------------------------------------------------------------------------
# Standalone functions for optimizer inspection
# ---------------------------------------------------------------------------


def is_optim_debug_iter(optimizer: torch.optim.Optimizer) -> bool:
    """Check if this iteration should run optimizer debug inspection.

    Args:
        optimizer: The optimizer to check.

    Returns:
        True if optimizer debug inspection should run this iteration.
    """
    MCoreDebugState.ensure_initialized()
    if not MCoreDebugState.debug_enabled:
        return False

    state = TensorInspectRegistry.get_optim_state(id(optimizer))
    return state.update_for_iteration(MCoreDebugState.get_iteration())


def inspect_optimizer_param(
    optimizer: torch.optim.Optimizer,
    param_name: str,
    param: torch.Tensor,
    grad: Optional[torch.Tensor],
    optimizer_state: Dict,
    iteration: int,
    reduction_group: Optional[torch.distributed.ProcessGroup] = None,
    is_distributed_optimizer: bool = False,
) -> None:
    """Inspect a parameter and collect optimizer statistics.

    Args:
        optimizer: The optimizer (used to key state).
        param_name: The parameter name.
        param: The parameter tensor.
        grad: The gradient tensor (may be None).
        optimizer_state: The optimizer state dict for this param.
        iteration: Current iteration number.
        reduction_group: Optional process group for reduction.
        is_distributed_optimizer: Whether this is a distributed optimizer.
    """
    import nvdlfw_inspect.api as debug_api

    state = TensorInspectRegistry.get_optim_state(id(optimizer))

    result = debug_api.megatron_core.inspect_optimizer_param_enabled(
        layer_name=param_name,
        param_name=param_name,
        iteration=iteration,
    )

    if isinstance(result, tuple):
        enabled, next_iter = result
        _update_next_debug_iter(state, next_iter, iteration)
    else:
        enabled = result

    if enabled:
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


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _update_next_debug_iter(
    state: Union[LayerDebugState, OptimizerDebugState],
    next_iter: Optional[int],
    current_iter: int,
) -> None:
    """Update next_debug_iter based on API response."""
    if next_iter is None:
        if state.next_debug_iter is None or state.next_debug_iter <= current_iter:
            state.next_debug_iter = None
    else:
        if state.next_debug_iter is None or state.next_debug_iter <= current_iter:
            state.next_debug_iter = next_iter
        else:
            state.next_debug_iter = min(state.next_debug_iter, next_iter)


def compute_next_enabled_iter(
    start_step: Optional[int],
    end_step: Optional[int],
    start_end_list: Optional[list],
    freq: int,
    current_iter: int,
) -> Tuple[bool, Optional[int]]:
    """Compute next enabled iteration based on scheduling parameters."""
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
