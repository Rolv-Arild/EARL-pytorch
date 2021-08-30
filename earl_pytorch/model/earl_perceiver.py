from typing import Optional

import torch
from torch import nn as nn
from torch.nn.modules.transformer import _get_activation_fn

from earl_pytorch.util.constants import DEFAULT_FEATURES


class EARLPerceiverBlock(nn.Module):
    def __init__(self, n_dims, n_heads, activation="relu", dim_feedforward=None):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 2 * n_dims
        self.attention = nn.MultiheadAttention(n_dims, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(n_dims)
        self.norm2 = nn.LayerNorm(n_dims)
        self.linear1 = nn.Linear(n_dims, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, n_dims)
        self.activation = _get_activation_fn(activation)

    def forward(self, src, invariant, mask=None):
        # Uses Pre-LN
        src2 = self.norm1(src)
        src2 = self.attention(src2, invariant, invariant, key_padding_mask=mask)[0]
        src = src + src2
        src2 = self.norm1(src)
        src2 = self.linear2(self.activation(self.linear1(src2)))
        src = src + src2
        return src


class EARLPerceiver(nn.Module):
    def __init__(
            self,
            n_dims: int = 256,
            n_layers: int = 2,
            n_heads: int = 4,
            n_preprocess_layers: int = 1,
            n_postprocess_layers: int = 0,
            n_features: Optional[int] = None
    ):
        """
        EARLPerceiver is an alternative to EARL that uses only a set number of embedding that attend to all the inputs.
        This reduces complexity from O(n^2) to O(n) and gives improved performance specifically on CPU.
        Note: it also uses a pre-LN block.

        :param n_dims: number of dimensions in the intermediate and output representations.
        :param n_layers: number of encoder layers.
        :param n_heads: number of heads in encoder layers.
        :param n_preprocess_layers: number of dense layers before encoder.
        :param n_postprocess_layers: number of dense layers after encoder.
        :param n_features: number of features in the input (last dimension).
        """
        super().__init__()
        self.n_dims = n_dims
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_features = n_features or len(DEFAULT_FEATURES)
        self.initial_dense = nn.Linear(self.n_features, n_dims)
        self.preprocess = nn.Sequential(*[nn.Linear(n_dims, n_dims) for _ in range(n_preprocess_layers)])

        self.blocks = nn.ModuleList([EARLPerceiverBlock(n_dims, n_heads) for _ in range(n_layers)])

        self.postprocess = nn.Sequential(*[nn.Linear(n_dims, n_dims) for _ in range(n_postprocess_layers)])

    def forward(self, query_entities: torch.Tensor, key_value_entities: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        q_emb = self.initial_dense(query_entities)
        kv_emb = self.initial_dense(key_value_entities)

        q_emb = self.preprocess(q_emb)
        kv_emb = self.preprocess(kv_emb)
        for block in self.blocks:
            q_emb = block(q_emb, kv_emb, mask=mask)
        q_emb = self.postprocess(q_emb)
        return q_emb