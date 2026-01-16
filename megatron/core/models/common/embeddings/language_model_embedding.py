# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from typing import Literal, Optional

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.debug.utils import TensorInspectMixin
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_tensor_model_parallel_group_if_none, nvtx_decorator


class LanguageModelEmbedding(MegatronModule, TensorInspectMixin):
    """Language model embeddings.

    Args:
        config (TransformerConfig): config object with all necessary configs for TransformerBlock
        vocab_size (int): vocabulary size
        max_sequence_length (int): maximum size of sequence. This
                             is used for positional embedding
        add_position_embedding (bool): Add a position embedding.
        embedding_dropout_prob (float): dropout probability for embeddings
        num_tokentypes (int): Set to 0 without binary head, and 2 with a binary head. Defaults to 0.
        scatter_to_sequence_parallel (bool): Set to False to disable scatter of embedding
            across sequence parallel region. Defaults to True.
    """

    def __init__(
        self,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
        num_tokentypes: int = 0,
        scatter_to_sequence_parallel: bool = True,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config
        self.vocab_size: int = vocab_size
        self.max_sequence_length: int = max_sequence_length
        self.add_position_embedding: bool = position_embedding_type == 'learned_absolute'
        self.num_tokentypes = num_tokentypes
        self.scatter_to_sequence_parallel = scatter_to_sequence_parallel
        self.tp_group = get_tensor_model_parallel_group_if_none(tp_group)
        self.reduce_scatter_embeddings = (
            (not self.add_position_embedding)
            and self.num_tokentypes <= 0
            and self.config.sequence_parallel
            and self.scatter_to_sequence_parallel
        )

        # Word embeddings (parallel).
        self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.config.hidden_size,
            init_method=self.config.embedding_init_method,
            reduce_scatter_embeddings=self.reduce_scatter_embeddings,
            config=self.config,
            tp_group=self.tp_group,
        )

        # Position embedding (serial).
        if self.add_position_embedding:
            self.position_embeddings = torch.nn.Embedding(
                self.max_sequence_length, self.config.hidden_size
            )

            # Initialize the position embeddings.
            if self.config.perform_initialization:
                self.config.embedding_init_method(self.position_embeddings.weight)

        if self.num_tokentypes > 0:
            self.tokentype_embeddings = torch.nn.Embedding(
                self.num_tokentypes, self.config.hidden_size
            )
            # Initialize the token-type embeddings.
            if self.config.perform_initialization:
                self.config.embedding_init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(self.config.hidden_dropout)

        self._setup_backward_hooks()

    def zero_parameters(self):
        """Zero out all parameters in embedding."""
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.shared = True
        self.position_embeddings.weight.data.fill_(0)
        self.position_embeddings.weight.shared = True
        if self.num_tokentypes > 0:
            self.tokentype_embeddings.weight.data.fill_(0)
            self.tokentype_embeddings.weight.shared = True

    def _get_debug_name(self) -> str:
        if self._debug_name is None:
            self._debug_name = "embedding"
        return self._debug_name

    def _get_reduction_group(self):
        return self.tp_group

    def _get_gradient_targets(self):
        return {"wgrad": self.word_embeddings, "dgrad": self}

    @nvtx_decorator()
    def forward(self, input_ids: Tensor, position_ids: Tensor, tokentype_ids: int = None) -> Tensor:
        """Forward pass of the embedding module.

        Args:
            input_ids (Tensor): The input tokens
            position_ids (Tensor): The position id's used to calculate position embeddings
            tokentype_ids (int): The token type ids. Used when args.bert_binary_head is
                set to True. Defaults to None

        Returns:
            Tensor: The output embeddings
        """
        word_embeddings = self.word_embeddings(input_ids)
        self._inspect_tensor("word", word_embeddings)

        if self.add_position_embedding:
            position_embeddings = self.position_embeddings(position_ids)
            self._inspect_tensor("position", position_embeddings)
            embeddings = word_embeddings + position_embeddings
        else:
            embeddings = word_embeddings

        if not self.reduce_scatter_embeddings:
            # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
            embeddings = embeddings.transpose(0, 1).contiguous()

        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            # [b s h] -> [s b h] (So that it can be added with embeddings)
            tokentype_embedding = self.tokentype_embeddings(tokentype_ids).permute(1, 0, 2)
            self._inspect_tensor("tokentype", tokentype_embedding)
            embeddings = embeddings + tokentype_embedding
        else:
            assert self.tokentype_embeddings is None

        self._inspect_tensor("output", embeddings)

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.config.fp32_residual_connection:
            embeddings = embeddings.float()

        # Dropout.
        if self.config.sequence_parallel:
            if not self.reduce_scatter_embeddings and self.scatter_to_sequence_parallel:
                embeddings = tensor_parallel.scatter_to_sequence_parallel_region(
                    embeddings, group=self.tp_group
                )
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            if self.config.clone_scatter_output_in_embedding and self.scatter_to_sequence_parallel:
                embeddings = embeddings.clone()
            with tensor_parallel.get_cuda_rng_tracker().fork():
                embeddings = self.embedding_dropout(embeddings)
        else:
            embeddings = self.embedding_dropout(embeddings)

        return embeddings
