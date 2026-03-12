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

This module provides standalone functions for tensor inspection
without requiring mixin inheritance in model classes.
"""

import fnmatch
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from megatron.core.debug.debug_state import MCoreDebugState


def matches_pattern(name: str, patterns: Optional[List[str]]) -> bool:
    """Return True if name matches any glob pattern in the list."""
    if not patterns or "*" in patterns:
        return True
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def build_options_tuple(config: dict) -> tuple:
    """Build the (start_step, end_step, start_end_list) options tuple from config."""
    start_step = config.get("start_step", None)
    end_step = config.get("end_step", None)
    start_end_list = config.get("start_end_list", None)
    if start_end_list is not None:
        start_end_list = tuple(
            tuple(int(x) for x in interval) for interval in start_end_list
        )
    return (start_step, end_step, start_end_list)


def _get_linear_types() -> tuple:
    """Build tuple of linear layer types to capture gradients from."""
    types: List[type] = [nn.Linear, nn.Embedding]

    try:
        from megatron.core.tensor_parallel.layers import (
            ColumnParallelLinear,
            RowParallelLinear,
        )
        types.extend([ColumnParallelLinear, RowParallelLinear])
    except ImportError:
        pass

    try:
        from megatron.core.extensions.transformer_engine import (
            TELinear,
            TEColumnParallelLinear,
            TERowParallelLinear,
            TELayerNormColumnParallelLinear,
        )
        types.extend([
            TELinear, TEColumnParallelLinear, TERowParallelLinear,
            TELayerNormColumnParallelLinear
        ])
    except ImportError:
        pass

    try:
        from megatron.core.extensions.transformer_engine import (
            TEGroupedLinear,
            TEColumnParallelGroupedLinear,
            TERowParallelGroupedLinear,
        )
        if TEGroupedLinear is not None:
            types.extend([
                TEGroupedLinear, TEColumnParallelGroupedLinear,
                TERowParallelGroupedLinear
            ])
    except ImportError:
        pass

    return tuple(types)


LINEAR_TYPES = _get_linear_types()


def get_reduction_params(
    tensor_name: str,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Tuple[bool, Optional[torch.distributed.ProcessGroup]]:
    """Get statistics reduction parameters for a tensor."""
    import nvdlfw_inspect.api as debug_api

    skip_reduction = False

    if tensor_name.lower() == "weight":
        if MCoreDebugState.weight_tensor_tp_group_reduce:
            reduction_group = tp_group
        else:
            skip_reduction = True
            reduction_group = None
    else:
        reduction_group = debug_api.get_tensor_reduction_group()

    return skip_reduction, reduction_group


@dataclass
class _BaseDebugState:
    """Base debug state for iteration gating."""

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


@dataclass
class LayerDebugState(_BaseDebugState):
    """Debug state for a single layer."""

    backward_hook_handles: Dict[str, torch.utils.hooks.RemovableHandle] = field(
        default_factory=dict
    )


class TensorInspectRegistry:
    """Central registry for per-layer debug state."""

    _layer_state: Dict[str, LayerDebugState] = {}

    @classmethod
    def get_layer_state(cls, layer_name: str) -> LayerDebugState:
        """Get or create debug state for a layer."""
        if layer_name not in cls._layer_state:
            cls._layer_state[layer_name] = LayerDebugState()
        return cls._layer_state[layer_name]

    @classmethod
    def reset(cls) -> None:
        """Reset all debug state."""
        for state in cls._layer_state.values():
            for handle in state.backward_hook_handles.values():
                handle.remove()
            state.backward_hook_handles.clear()
        cls._layer_state.clear()


def is_debug_iter(layer_name: str) -> bool:
    """Check if this iteration should run tensor debug inspection for the layer."""
    MCoreDebugState.ensure_initialized()
    if not MCoreDebugState.debug_enabled:
        state = TensorInspectRegistry.get_layer_state(layer_name)
        if state.backward_hook_handles:
            remove_backward_hooks(layer_name)
        return False

    state = TensorInspectRegistry.get_layer_state(layer_name)
    return state.update_for_iteration(MCoreDebugState.get_iteration())


def _inspect_tensor_common(
    layer_name: str,
    tensor_name: str,
    tensor: torch.Tensor,
    reduction_group: Optional[torch.distributed.ProcessGroup],
    *,
    is_backward: bool,
) -> None:
    """Common gating and routing for forward and backward tensor inspection."""
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

    if not enabled:
        return

    if is_backward:
        dp_reduction_group = debug_api.get_tensor_reduction_group()
        debug_api.megatron_core.inspect_tensor(
            layer_name=layer_name,
            tensor_name=tensor_name,
            tensor=tensor,
            iteration=iteration,
            reduction_group=dp_reduction_group,
            skip_reduction=dp_reduction_group is None,
            tp_group=reduction_group,
        )
    else:
        skip_reduction, effective_reduction_group = get_reduction_params(
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


def inspect_tensor(
    layer_name: str,
    tensor_name: str,
    tensor: torch.Tensor,
    reduction_group: Optional[torch.distributed.ProcessGroup] = None,
) -> None:
    """Inspect a tensor and collect statistics."""
    _inspect_tensor_common(
        layer_name, tensor_name, tensor, reduction_group, is_backward=False
    )


def inspect_backward_tensor(
    layer_name: str,
    tensor_name: str,
    grad: torch.Tensor,
    reduction_group: Optional[torch.distributed.ProcessGroup] = None,
) -> None:
    """Inspect a backward tensor (wgrad/dgrad) and collect statistics."""
    _inspect_tensor_common(
        layer_name, tensor_name, grad, reduction_group, is_backward=True
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
    """Remove all registered backward hooks for a layer."""
    state = TensorInspectRegistry.get_layer_state(layer_name)
    for handle in state.backward_hook_handles.values():
        handle.remove()
    state.backward_hook_handles.clear()


def _unwrap_model(model: Any) -> Any:
    """Unwrap a model from DDP/FSDP wrappers."""
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    return unwrapped


def register_global_backward_hooks(
    model: Any,
    hook_factory: Callable[[str], Callable],
    layer_patterns: Optional[List[str]] = None,
    module_types: Optional[tuple] = None,
) -> List[torch.utils.hooks.RemovableHandle]:
    """Register backward hooks on matching modules.

    Args:
        model: The model or list of model chunks to register hooks on.
        hook_factory: Callable that takes layer_name and returns a hook function.
        layer_patterns: Optional list of glob patterns to filter layers.
        module_types: Optional tuple of module types to register hooks on.

    Returns:
        List of hook handles that can be used to remove the hooks later.
    """
    if module_types is None:
        module_types = LINEAR_TYPES

    handles: List[torch.utils.hooks.RemovableHandle] = []
    model_chunks = model if isinstance(model, (list, tuple)) else [model]

    for chunk_id, model_chunk in enumerate(model_chunks):
        unwrapped = _unwrap_model(model_chunk)

        for module_name, module in unwrapped.named_modules():
            if isinstance(module, module_types):
                if matches_pattern(module_name, layer_patterns):
                    layer_name = f"model_chunk{chunk_id}__{module_name}"
                    hook_fn = hook_factory(layer_name)
                    handle = module.register_full_backward_hook(hook_fn)
                    handles.append(handle)

    return handles


def remove_hook_handles(handles: List[torch.utils.hooks.RemovableHandle]) -> None:
    """Remove a list of hook handles and clear the list."""
    for handle in handles:
        handle.remove()
    handles.clear()


def manage_backward_hooks(
    layer_name: str,
    gradient_targets: Dict[str, torch.nn.Module],
    reduction_group: Optional[torch.distributed.ProcessGroup] = None,
) -> None:
    """Dynamically manage backward hooks based on debug state.

    Call this at the start of forward() to set up or remove hooks as needed.
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


def _update_next_debug_iter(
    state: LayerDebugState,
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
    iteration: int,
) -> Tuple[bool, Optional[int]]:
    """Compute whether to run at this iteration and when next to run.

    Mirrors TE's next_enabled_iter() pattern for consistency across the stack.

    Returns:
        run_current: True if the feature should be enabled at the current iteration.
        next_iter: The next iteration when the feature will be enabled, or None.
    """
    run_current = False

    if start_end_list:
        intervals = sorted(start_end_list)
    else:
        start = 0 if start_step is None else start_step
        end = float("inf") if end_step is None else end_step
        intervals = [(start, end)]

    for s, e in intervals:
        if iteration % freq == 0 and s <= iteration <= e:
            run_current = True

        first = max(iteration + 1, s)
        offset = first % freq
        candidate = first if offset == 0 else first + (freq - offset)
        if candidate <= e:
            return run_current, candidate

    return run_current, None
