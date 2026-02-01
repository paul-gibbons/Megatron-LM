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

import fnmatch
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from megatron.core.debug.debug_state import MCoreDebugState


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
) -> Tuple[bool, Optional[torch.distributed.ProcessGroup], bool]:
    """Get statistics reduction parameters for a tensor."""
    import nvdlfw_inspect.api as debug_api

    skip_reduction = False
    reduce_within_microbatch = tensor_name.lower() != "weight"

    if tensor_name.lower() == "weight":
        if MCoreDebugState.weight_tensor_tp_group_reduce:
            reduction_group = tp_group
        else:
            skip_reduction = True
            reduction_group = None
    else:
        reduction_group = debug_api.get_tensor_reduction_group()

    return skip_reduction, reduction_group, reduce_within_microbatch


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


def _unwrap_model(model: Any) -> Any:
    """Unwrap a model from DDP/FSDP wrappers."""
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    return unwrapped


def _matches_layer_pattern(name: str, patterns: Optional[List[str]]) -> bool:
    """Return True if name matches any pattern."""
    if not patterns or "*" in patterns:
        return True
    return any(fnmatch.fnmatch(name, p) or p == "*" for p in patterns)


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
                if _matches_layer_pattern(module_name, layer_patterns):
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


_IS_OPTIM_DEBUG_ITER_LOGGED = {}  # Track logging per iteration


def is_optim_debug_iter(optimizer: torch.optim.Optimizer) -> bool:
    """Check if this iteration should run optimizer debug inspection.

    Args:
        optimizer: The optimizer to check.

    Returns:
        True if optimizer debug inspection should run this iteration.
    """
    import logging
    _logger = logging.getLogger(__name__)

    MCoreDebugState.ensure_initialized()

    current_iter = MCoreDebugState.get_iteration()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    # Debug log once per iteration
    should_log = current_iter not in _IS_OPTIM_DEBUG_ITER_LOGGED
    if should_log:
        _IS_OPTIM_DEBUG_ITER_LOGGED[current_iter] = True
        # Keep dict from growing unbounded
        if len(_IS_OPTIM_DEBUG_ITER_LOGGED) > 100:
            old_keys = sorted(_IS_OPTIM_DEBUG_ITER_LOGGED.keys())[:-50]
            for k in old_keys:
                del _IS_OPTIM_DEBUG_ITER_LOGGED[k]

    if not MCoreDebugState.debug_enabled:
        if should_log:
            _logger.info(
                f"[is_optim_debug_iter DEBUG] rank={rank} iter={current_iter} "
                f"SKIP - debug_enabled=False"
            )
        return False

    state = TensorInspectRegistry.get_optim_state(id(optimizer))
    result = state.update_for_iteration(current_iter)

    if should_log:
        _logger.info(
            f"[is_optim_debug_iter DEBUG] rank={rank} iter={current_iter} "
            f"result={result} next_debug_iter={state.next_debug_iter} "
            f"enabled_this_iter={state.enabled_this_iter}"
        )

    return result


def _infer_optimizer_type(optimizer: torch.optim.Optimizer) -> Optional[str]:
    """Infer optimizer type from class name for logging."""
    opt_class = type(optimizer).__name__.lower()
    if "muon" in opt_class:
        return "muon"
    if "adam" in opt_class or "fusedadam" in opt_class:
        return "adam"
    if "sgd" in opt_class:
        return "sgd"
    return None


_INSPECT_OPT_PARAM_DEBUG_LOGGED_ITER = -1  # Module-level tracker for debug logging


def inspect_optimizer_param(
    optimizer: torch.optim.Optimizer,
    param_name: str,
    param: torch.Tensor,
    grad: Optional[torch.Tensor],
    optimizer_state: Dict,
    iteration: int,
    reduction_group: Optional[torch.distributed.ProcessGroup] = None,
    is_distributed_optimizer: bool = False,
    optimizer_type: Optional[str] = None,
    is_expert_parallel: Optional[bool] = None,
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
        optimizer_type: Optional optimizer type name (e.g., "muon", "adam").
            If None, will be inferred from optimizer class name.
        is_expert_parallel: Whether the parameter is expert-parallel.
    """
    import logging
    import nvdlfw_inspect.api as debug_api

    logger = logging.getLogger(__name__)

    # Debug logging - once per iteration
    global _INSPECT_OPT_PARAM_DEBUG_LOGGED_ITER
    debug_this_call = (_INSPECT_OPT_PARAM_DEBUG_LOGGED_ITER != iteration)
    if debug_this_call:
        _INSPECT_OPT_PARAM_DEBUG_LOGGED_ITER = iteration
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        reduction_group_info = "None"
        if reduction_group is not None:
            try:
                reduction_group_info = f"size={torch.distributed.get_world_size(reduction_group)}"
            except Exception:
                reduction_group_info = "error"
        logger.info(
            f"[inspect_optimizer_param DEBUG] rank={rank} iter={iteration}\n"
            f"  param_name={param_name}\n"
            f"  param.shape={param.shape}\n"
            f"  grad={'None' if grad is None else f'shape={grad.shape}'}\n"
            f"  is_distributed_optimizer={is_distributed_optimizer}\n"
            f"  reduction_group={reduction_group_info}\n"
            f"  is_expert_parallel={is_expert_parallel}"
        )

    # Auto-detect optimizer type if not provided
    if optimizer_type is None:
        optimizer_type = _infer_optimizer_type(optimizer)

    state = TensorInspectRegistry.get_optim_state(id(optimizer))

    result = debug_api.megatron_core.inspect_optimizer_param_enabled(
        layer_name=param_name,
        param_name=param_name,
        iteration=iteration,
        is_expert_parallel=is_expert_parallel,
    )

    if isinstance(result, tuple):
        enabled, next_iter = result
        _update_next_debug_iter(state, next_iter, iteration)
    else:
        enabled = result

    if debug_this_call:
        logger.info(
            f"[inspect_optimizer_param DEBUG] rank={rank} iter={iteration} "
            f"enabled={enabled} for param={param_name}"
        )

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
            optimizer_type=optimizer_type,
            is_expert_parallel=is_expert_parallel,
        )


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
