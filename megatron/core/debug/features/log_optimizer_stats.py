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

"""LogOptimizerStats feature for optimizer statistics."""
from typing import Dict, Optional, Tuple

import torch

from nvdlfw_inspect.logging import MetricLogger
from nvdlfw_inspect.registry import Registry, api_method

from megatron.core.debug.features.api import MCoreConfigAPIMapper
from megatron.core.debug.features.utils.optimizer_stats_buffer import OPTIMIZER_STATS_BUFFERS
from megatron.core.debug.utils import compute_next_enabled_iter


@Registry.register_feature(namespace="megatron_core")
class LogOptimizerStats(MCoreConfigAPIMapper):
    """Log optimizer statistics per parameter or aggregated by layer.

    Supported stats:
        num_zeros, num_zeros%, num_zeros[threshold]%, grad_norm, grad_rms,
        weight_grad_ratio, exp_avg_norm, exp_avg_sq_mean, rms_staleness, update_norm
    """

    _SUPPORTED_STATS = {
        "num_zeros", "num_zeros%", "grad_norm", "grad_rms", "weight_grad_ratio",
        "exp_avg_norm", "exp_avg_sq_mean", "grad_to_v_ratio", "rms_staleness", "update_norm",
        "per_token_grad_norm",
    }

    def _check_log_frequency(self, config: Dict, iteration: int) -> Tuple[bool, Optional[int]]:
        return compute_next_enabled_iter(
            config.get("start_step", 0),
            config.get("end_step", -1),
            config.get("start_end_list"),
            config.get("freq", 1),
            iteration,
        )

    def _validate_stats(self, stats: list) -> None:
        from megatron.core.debug.features.utils.stats_buffer import parse_num_zeros_stat
        for stat in stats:
            if parse_num_zeros_stat(stat) is None and stat.lower() not in self._SUPPORTED_STATS:
                raise ValueError(f"Unsupported optimizer stat: '{stat}'")

    def _get_per_token_topk_list(self, config: Dict) -> list[int]:
        topk = config.get("per_token_topk", [1, 3, 7, 10, 25, 50, 100])
        if not isinstance(topk, (list, tuple)) or not topk:
            raise ValueError("[MCore Debug] per_token_topk must be a non-empty list of ints.")
        topk_int = []
        for k in topk:
            k_int = int(k)
            if k_int <= 0:
                raise ValueError("[MCore Debug] per_token_topk entries must be > 0.")
            topk_int.append(k_int)
        return sorted(set(topk_int))

    def _maybe_get_tp_group_and_rank(self):
        """Best-effort access to tensor-parallel group for vocab-sharded weights."""
        try:
            from megatron.core import parallel_state

            if not parallel_state.model_parallel_is_initialized():
                return None, 0, 1
            tp_group = parallel_state.get_tensor_model_parallel_group()
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            tp_world = parallel_state.get_tensor_model_parallel_world_size()
            return tp_group, tp_rank, tp_world
        except Exception:
            return None, 0, 1

    def _maybe_gather_dp_shards(
        self,
        grad: Optional[torch.Tensor],
        reduction_group: Optional[torch.distributed.ProcessGroup],
        is_distributed_optimizer: bool,
    ) -> Optional[torch.Tensor]:
        if grad is None or not is_distributed_optimizer:
            return grad
        if reduction_group is None or not torch.distributed.is_initialized():
            return grad
        world_size = torch.distributed.get_world_size(reduction_group)
        if world_size <= 1 or grad.dim() != 1:
            return grad

        local_numel = torch.tensor([grad.numel()], device=grad.device, dtype=torch.int64)
        gathered_sizes = [torch.empty_like(local_numel) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_sizes, local_numel, group=reduction_group)
        sizes = [int(t.item()) for t in gathered_sizes]
        if sum(sizes) == 0:
            return grad

        max_size = max(sizes)
        if grad.numel() == max_size:
            grad_padded = grad
        else:
            grad_padded = torch.zeros(max_size, device=grad.device, dtype=grad.dtype)
            grad_padded[: grad.numel()] = grad

        gathered = [torch.empty_like(grad_padded) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, grad_padded, group=reduction_group)
        return torch.cat([g[:s] for g, s in zip(gathered, sizes)], dim=0)

    def _infer_per_token_shape(
        self,
        grad: torch.Tensor,
        param: Optional[torch.Tensor],
    ) -> Optional[Tuple[int, int]]:
        numel = grad.numel()

        def shape_from_hidden(size: Optional[int]) -> Optional[Tuple[int, int]]:
            if isinstance(size, int) and size > 0 and numel % size == 0:
                return (numel // size, size)
            return None

        def shape_from_vocab(size: Optional[int]) -> Optional[Tuple[int, int]]:
            if isinstance(size, int) and size > 0 and numel % size == 0:
                return (size, numel // size)
            return None

        args_hidden_size = None
        args_vocab_size = None
        try:
            from megatron.training.global_vars import get_args as _get_args

            _args = _get_args()
            args_hidden_size = getattr(_args, "hidden_size", None)
            args_vocab_size = getattr(_args, "padded_vocab_size", None) or getattr(
                _args, "vocab_size", None
            )
        except Exception:
            pass

        shape = shape_from_hidden(args_hidden_size) or shape_from_vocab(args_vocab_size)
        if shape is not None:
            return shape

        if (
            isinstance(param, torch.Tensor)
            and param.dim() >= 2
            and numel == param.numel()
        ):
            return tuple(param.shape)
        return None

    def _log_per_token_grad_norm(
        self,
        *,
        config: Dict,
        param_name: str,
        grad: Optional[torch.Tensor],
        param: Optional[torch.Tensor],
        iteration: int,
        reduction_group: Optional[torch.distributed.ProcessGroup],
        is_distributed_optimizer: bool,
    ) -> None:
        """Compute per-token (vocab-row) grad L2 norms and top-k contributions."""
        grad = self._maybe_gather_dp_shards(grad, reduction_group, is_distributed_optimizer)
        if grad is None or grad.numel() == 0:
            return

        if grad.dim() < 2:
            target_shape = self._infer_per_token_shape(grad, param)
            if target_shape is None:
                return
            grad = grad.view(*target_shape)

        g = grad.detach()
        g2 = g.float().pow(2)
        reduce_dims = tuple(range(1, g2.dim()))
        row_norm_sq = g2.sum(dim=reduce_dims)  # [dim0]
        row_norm = row_norm_sq.sqrt()

        local_rows = int(row_norm.numel())
        if local_rows == 0:
            return

        tp_group, tp_rank, tp_world = self._maybe_get_tp_group_and_rank()

        token_offset = tp_rank * local_rows
        if tp_group is not None and torch.distributed.is_initialized() and tp_world > 1:
            rows_t = torch.tensor([local_rows], device=row_norm.device, dtype=torch.int64)
            gathered_rows = [torch.empty_like(rows_t) for _ in range(tp_world)]
            torch.distributed.all_gather(gathered_rows, rows_t, group=tp_group)
            counts = [int(t.item()) for t in gathered_rows]
            token_offset = sum(counts[:tp_rank])

        local_token_ids = torch.arange(local_rows, device=row_norm.device, dtype=torch.int64) + token_offset
        topk_list = self._get_per_token_topk_list(config)
        k_max = min(max(topk_list), local_rows)
        local_vals, local_idx = torch.topk(row_norm_sq, k=k_max, largest=True, sorted=True)
        local_ids_top = local_token_ids.index_select(0, local_idx)

        if tp_group is not None and torch.distributed.is_initialized() and tp_world > 1:
            total_sq = row_norm_sq.sum(dtype=torch.float64)
            torch.distributed.all_reduce(total_sq, op=torch.distributed.ReduceOp.SUM, group=tp_group)

            sum_norm = row_norm.sum(dtype=torch.float64)
            torch.distributed.all_reduce(sum_norm, op=torch.distributed.ReduceOp.SUM, group=tp_group)
            count = torch.tensor([local_rows], device=row_norm.device, dtype=torch.float64)
            torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM, group=tp_group)
            mean_norm = (sum_norm / count).item() if count.item() > 0 else 0.0

            median_norm = None
            if bool(config.get("per_token_compute_median", True)):
                gathered_norm = [torch.empty_like(row_norm) for _ in range(tp_world)]
                torch.distributed.all_gather(gathered_norm, row_norm, group=tp_group)
                median_norm = torch.cat(gathered_norm, dim=0).median().item()

            gathered_vals = [torch.empty_like(local_vals) for _ in range(tp_world)]
            gathered_ids = [torch.empty_like(local_ids_top) for _ in range(tp_world)]
            torch.distributed.all_gather(gathered_vals, local_vals, group=tp_group)
            torch.distributed.all_gather(gathered_ids, local_ids_top, group=tp_group)

            cand_vals = torch.cat(gathered_vals, dim=0)
            cand_ids = torch.cat(gathered_ids, dim=0)
            global_vals, global_pos = torch.topk(cand_vals, k=min(max(topk_list), cand_vals.numel()), largest=True, sorted=True)
            global_ids = cand_ids.index_select(0, global_pos)

            prefix = f"optimizer_{param_name}_per_token_grad_norm"
            MetricLogger.log_scalar(f"{prefix}_mean", float(mean_norm), iteration)
            if median_norm is not None:
                MetricLogger.log_scalar(f"{prefix}_median", float(median_norm), iteration)
            max_norm = float(global_vals[0].sqrt().item()) if global_vals.numel() > 0 else 0.0
            MetricLogger.log_scalar(f"{prefix}_max", max_norm, iteration)

            denom = float(total_sq.item()) if total_sq.item() > 0 else 1.0
            for k in topk_list:
                k_eff = min(k, int(global_vals.numel()))
                pct = 100.0 * float(global_vals[:k_eff].sum().item()) / denom
                MetricLogger.log_scalar(f"{prefix}_top{k_eff}_pct", pct, iteration)

            topn = int(config.get("per_token_log_topn", 10))
            topn = max(0, min(topn, int(global_vals.numel())))
            for i in range(topn):
                token_id = float(global_ids[i].item())
                token_norm = float(global_vals[i].sqrt().item())
                MetricLogger.log_scalar(f"{prefix}_top{i}_token_id", token_id, iteration)
                MetricLogger.log_scalar(f"{prefix}_top{i}_token_norm", token_norm, iteration)

        else:
            prefix = f"optimizer_{param_name}_per_token_grad_norm"
            MetricLogger.log_scalar(f"{prefix}_mean", float(row_norm.mean().item()), iteration)
            MetricLogger.log_scalar(f"{prefix}_median", float(row_norm.median().item()), iteration)
            MetricLogger.log_scalar(f"{prefix}_max", float(row_norm.max().item()), iteration)
            total_sq = row_norm_sq.sum(dtype=torch.float64)
            denom = float(total_sq.item()) if total_sq.item() > 0 else 1.0
            for k in topk_list:
                k_eff = min(k, int(local_vals.numel()))
                pct = 100.0 * float(local_vals[:k_eff].sum().item()) / denom
                MetricLogger.log_scalar(f"{prefix}_top{k_eff}_pct", pct, iteration)

            topn = int(config.get("per_token_log_topn", 10))
            topn = max(0, min(topn, int(local_vals.numel())))
            for i in range(topn):
                token_id = float(local_ids_top[i].item())
                token_norm = float(local_vals[i].sqrt().item())
                MetricLogger.log_scalar(f"{prefix}_top{i}_token_id", token_id, iteration)
                MetricLogger.log_scalar(f"{prefix}_top{i}_token_norm", token_norm, iteration)

    @api_method
    def inspect_optimizer_param_enabled(
        self, config: Dict, layer_name: str, iteration: int, **kwargs
    ) -> Tuple[bool, Optional[int]]:
        return self._check_log_frequency(config, iteration)

    @api_method
    def inspect_optimizer_param(
        self, config: Dict, layer_name: str, param: torch.Tensor, iteration: int, **kwargs
    ) -> None:
        param_name = kwargs.get("param_name", layer_name)
        grad = kwargs.get("grad")
        stats = config.get("stats", ["num_zeros%", "grad_norm"])

        should_run, _ = self._check_log_frequency(config, iteration)
        if not should_run:
            return

        self._validate_stats(stats)

        has_per_token = any(s.lower() == "per_token_grad_norm" for s in stats)
        if has_per_token:
                if config.get("aggregate_by") is not None:
                    raise ValueError(
                        "[MCore Debug] per_token_grad_norm does not support aggregate_by; "
                        "target a single parameter (e.g., output_layer.weight) via layer_name_regex_pattern."
                    )
                self._log_per_token_grad_norm(
                    config=config,
                    param_name=param_name,
                    grad=grad,
                param=param,
                    iteration=iteration,
                reduction_group=kwargs.get("reduction_group"),
                is_distributed_optimizer=bool(kwargs.get("is_distributed_optimizer", False)),
                )

        buffer_stats = [s for s in stats if s.lower() != "per_token_grad_norm"]
        if buffer_stats:
            OPTIMIZER_STATS_BUFFERS.feed(
                param_name=param_name,
                param=param,
                grad=grad,
                optimizer_state=kwargs.get("optimizer_state", {}),
                stats=buffer_stats,
                iteration=iteration,
                reduction_group=kwargs.get("reduction_group"),
                aggregate_by=config.get("aggregate_by"),
            )
