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

"""DGrad logger for capturing activation gradients."""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Any

import torch

from megatron.core.debug.utils import (
    LINEAR_TYPES,
    register_global_backward_hooks,
    remove_hook_handles,
)
from megatron.core.debug.features.utils.tensor_dump import (
    save_tensor_direct,
    _should_save,
)

logger = logging.getLogger(__name__)


class DGradLogger:
    """Captures and saves data gradients from linear layers using backward hooks."""

    def __init__(self):
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._dgrads: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        self._save_dir: Optional[str] = None
        self._iteration: Optional[int] = None
        self._enabled = False
        self._model: Optional[Any] = None
        self._layer_patterns: List[str] = ["*"]
        self._microbatch_indices: Dict[str, int] = {}

    def reset(self):
        self._dgrads.clear()
        self._microbatch_indices.clear()

    def _get_microbatch_index(self, layer_name: str) -> int:
        idx = self._microbatch_indices.get(layer_name, 0)
        self._microbatch_indices[layer_name] = idx + 1
        return idx

    def set_model(self, model: Any, layer_patterns: Optional[List[str]] = None):
        self._model = model
        self._layer_patterns = layer_patterns or ["*"]

    def enable(self, save_dir: str, iteration: int):
        if self._enabled and self._iteration == iteration:
            return

        self._save_dir = save_dir
        self._iteration = iteration
        self._enabled = True

        if self._model is not None and not self._hooks:
            self._register_hooks_internal()

    def disable(self):
        self._enabled = False
        self._remove_hooks_internal()

    def _make_hook_factory(self):
        return self._make_hook

    def _make_hook(self, layer_name: str):
        def hook(_module, grad_input, grad_output):
            if not self._enabled:
                return
            mb_idx = self._get_microbatch_index(layer_name)
            for idx, grad in enumerate(grad_output):
                if grad is not None:
                    key = f"dgrad_output{idx}__mb{mb_idx:02d}"
                    self._dgrads[layer_name][key] = grad.detach().cpu()
            for idx, grad in enumerate(grad_input):
                if grad is not None:
                    key = f"dgrad_input{idx}__mb{mb_idx:02d}"
                    self._dgrads[layer_name][key] = grad.detach().cpu()
        return hook

    def _register_hooks_internal(self):
        if self._model is None:
            return

        self._hooks = register_global_backward_hooks(
            model=self._model,
            hook_factory=self._make_hook_factory(),
            layer_patterns=self._layer_patterns,
            module_types=LINEAR_TYPES,
        )

    def _remove_hooks_internal(self):
        remove_hook_handles(self._hooks)

    def register_hooks(self, model: Any, layer_patterns: Optional[List[str]] = None):
        self.set_model(model, layer_patterns)

    def remove_hooks(self):
        self._remove_hooks_internal()
        self._model = None

    def save(self):
        self._remove_hooks_internal()
        self._enabled = False

        if not self._dgrads or not self._save_dir or self._iteration is None:
            return

        if not _should_save():
            self.reset()
            return

        dgrad_save_dir = f"{self._save_dir}/dgrads"
        for layer_name, tensors in self._dgrads.items():
            for tensor_name, tensor in tensors.items():
                save_tensor_direct(
                    dgrad_save_dir, self._iteration, layer_name, tensor_name, tensor,
                    log_to_metrics=False,
                    include_microbatch=False,
                    track_state=False,
                )

        count = sum(len(t) for t in self._dgrads.values())
        logger.info(f"[DGradLogger] Iteration {self._iteration}: saved {count} dgrads")
        self.reset()

    @property
    def has_data(self) -> bool:
        return bool(self._dgrads)

    @property
    def has_model(self) -> bool:
        return self._model is not None


DGRAD_LOGGER = DGradLogger()


def register_dgrad_hooks(model: Any, layer_patterns: List[str] = None):
    DGRAD_LOGGER.register_hooks(model, layer_patterns)


def remove_dgrad_hooks():
    DGRAD_LOGGER.remove_hooks()


def save_dgrads():
    DGRAD_LOGGER.save()


def enable_dgrad_capture(save_dir: str, iteration: int):
    DGRAD_LOGGER.enable(save_dir, iteration)


def disable_dgrad_capture():
    DGRAD_LOGGER.disable()
