from typing import Optional

import torch
from torch import nn as nn
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F

from ..util.constants import DEFAULT_FEATURES
from ..util.util import mlp


class EARLPerceiverBlock(nn.Module):
    def __init__(self, n_dims, n_heads, activation=F.relu, dim_feedforward=None, concatenate=False, use_norm=True):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * n_dims
        self.concatenate = concatenate

        # Layers
        self.attention = nn.MultiheadAttention(n_dims, n_heads, batch_first=True)
        self.linear1 = nn.Linear((1 + self.concatenate) * n_dims, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, n_dims)

        self.activation = activation

        self.use_norm = use_norm
        if use_norm:
            self.norm1 = nn.LayerNorm(n_dims)
            self.norm2 = nn.LayerNorm(n_dims)
            self.norm3 = nn.LayerNorm(n_dims)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            self.norm3 = nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model. Taken from PyTorch Transformer impl"""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, invariant, mask=None):
        src2 = self.norm1(src)
        invariant = self.norm2(invariant)
        src2, weights = self.attention(src2, invariant, invariant, key_padding_mask=mask)

        src = torch.cat((src, src2), dim=-1) if self.concatenate else (src + src2)

        src2 = self.norm3(src)
        src2 = self.linear2(self.activation(self.linear1(src2)))

        # src = torch.cat((src, src2), dim=-1) if self.concatenate else (src + src2)
        src = src + src2

        return src, weights


class EARLPerceiver(nn.Module):
    def __init__(
            self,
            n_dims: int = 256,
            n_layers: int = 2,
            n_heads: int = 4,
            n_preprocess_layers: int = 1,
            n_postprocess_layers: int = 0,
            query_features: Optional[int] = None,
            key_value_features: Optional[int] = None,
            return_weights=False,
            activation=F.relu
    ):
        """
        EARLPerceiver is an alternative to EARL that uses only a set number of embedding that attend to all the inputs.
        This reduces complexity from O(n^2) to O(n) and gives improved performance specifically on CPU.

        :param n_dims: number of dimensions in the intermediate and output representations.
        :param n_layers: number of encoder layers.
        :param n_heads: number of heads in encoder layers.
        :param n_preprocess_layers: number of dense layers before encoder.
        :param n_postprocess_layers: number of dense layers after encoder.
        :param query_features: number of features in the query input (last dimension).
        :param key_value_features: number of features in the key_value input (last dimension).
        """
        super().__init__()
        if query_features is None:
            query_features = len(DEFAULT_FEATURES)
        if key_value_features is None:
            key_value_features = len(DEFAULT_FEATURES)

        self.n_dims = n_dims
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.return_weights = return_weights

        if n_preprocess_layers > 0:
            self.query_preprocess = mlp(query_features, n_dims, n_preprocess_layers)
            self.key_value_preprocess = mlp(key_value_features, n_dims, n_preprocess_layers)
        else:
            self.query_preprocess = nn.Identity()
            self.key_value_preprocess = nn.Identity()

        self.blocks = nn.ModuleList([
            EARLPerceiverBlock(n_dims, n_heads, activation=activation)
            for _ in range(n_layers)
        ])

        if n_postprocess_layers > 0:
            self.postprocess = mlp(n_dims, n_dims, n_postprocess_layers - 1, n_dims)
        else:
            self.postprocess = nn.Identity()

    def forward(self, query_entities: torch.Tensor, key_value_entities: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        q_emb = self.query_preprocess(query_entities)
        kv_emb = self.key_value_preprocess(key_value_entities)

        weights = []
        for block in self.blocks:
            q_emb, w = block(q_emb, kv_emb, mask=mask)
            weights.append(w)

        q_emb = self.postprocess(q_emb)

        if self.return_weights:
            return q_emb, tuple(weights)
        return q_emb
