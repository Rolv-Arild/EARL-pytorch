from typing import Optional

import torch
from torch import nn as nn
from torch.nn.init import xavier_uniform_

from ..util.constants import DEFAULT_FEATURES
from ..util.util import mlp


class EARL(nn.Module):
    def __init__(
            self,
            n_dims: int = 128,
            n_layers: int = 4,
            n_heads: int = 4,
            n_preprocess_layers: int = 1,
            n_postprocess_layers: int = 0,
            dropout: float = 0.,
            n_features: Optional[int] = None,
            dim_feedforward: Optional[int] = None
    ):
        """
        Create an EARL (Extensible Attention-based Rocket League) model

        :param n_dims: number of dimensions in the intermediate and output representations.
        :param n_layers: number of encoder layers.
        :param n_heads: number of heads in encoder layers.
        :param n_preprocess_layers: number of dense layers before encoder.
        :param n_postprocess_layers: number of dense layers after encoder.
        :param dropout: dropout for encoder layers.
        :param n_features: number of features in the input (last dimension).
        """
        super().__init__()
        self.n_features = n_features or len(DEFAULT_FEATURES)
        self.n_dims = n_dims
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.preprocess = mlp(self.n_features, n_dims, n_preprocess_layers)
        if n_postprocess_layers > 0:
            self.postprocess = mlp(n_dims, n_dims, n_postprocess_layers - 1, n_dims)
        else:
            self.postprocess = nn.Identity()

        if dim_feedforward is None:
            dim_feedforward = 2 * n_dims

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(n_dims, n_heads, dim_feedforward, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model. Taken from PyTorch Transformer impl"""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, entities: torch.Tensor, mask: Optional[torch.Tensor] = None):
        emb = self.initial_dense(entities)
        emb = self.preprocess(emb)
        for block in self.blocks:
            emb = block(emb, src_key_padding_mask=mask)
        emb = self.postprocess(emb)
        return emb

    def __repr__(self):
        return f"EARL(n_dims={self.n_dims},n_layers={self.n_layers},n_heads={self.n_heads})"
