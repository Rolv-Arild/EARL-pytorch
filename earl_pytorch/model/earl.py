from typing import overload, Tuple, Optional

import torch
from torch import nn as nn

DEFAULT_N_PARAMETERS = 21


class EARL(nn.Module):
    def __init__(self, n_dims=128, n_layers=4, n_heads=4, dropout=0.1, encoder_norm=True, n_parameters=None,
                 include_cls=True):
        super().__init__()
        self.n_parameters = n_parameters or DEFAULT_N_PARAMETERS
        self.n_dims = n_dims
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.include_cls = include_cls

        if include_cls:
            self.emb = nn.Parameter(torch.Tensor(self.n_parameters), requires_grad=True)
            nn.init.normal_(self.emb)
        self.initial_dense = nn.Linear(self.n_parameters, n_dims)
        self.final_dense = nn.Linear(n_dims, n_dims)
        encoder_layer = nn.TransformerEncoderLayer(n_dims, n_heads, n_dims, dropout=dropout)
        if encoder_norm:
            encoder_norm = nn.LayerNorm(n_dims)
        else:
            encoder_norm = None

        self.blocks = nn.TransformerEncoder(
            encoder_layer, n_layers, encoder_norm
        )

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model. From PyTorch's transformer implementation."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_categories(self, balls: torch.Tensor, boosts: torch.Tensor, players: torch.Tensor,
                           mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        entities = torch.cat((balls, boosts, players), dim=1)
        if self.include_cls:
            cls_emb, entity_emb = self.forward_entities(entities, mask=mask)
        else:
            entity_emb = self.forward_entities(entities, mask=mask)

        n_balls = balls.shape[1]
        n_boosts = boosts.shape[1]
        n_players = players.shape[1]

        ball_emb = entity_emb.narrow(1, 0, n_balls)
        boost_emb = entity_emb.narrow(1, n_balls, n_boosts)
        player_emb = entity_emb.narrow(1, n_balls + n_boosts, n_players)

        if self.include_cls:
            return cls_emb, ball_emb, boost_emb, player_emb
        else:
            return ball_emb, boost_emb, player_emb

    def forward_entities(self, entities: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        if self.include_cls:
            cls = self.emb.repeat(entities.shape[0], 1, 1)
            conc = torch.cat((cls, entities), dim=1)
        else:
            conc = entities

        emb = self.initial_dense(conc)
        emb = emb.swapdims(0, 1)
        emb = self.blocks(emb, mask=mask)
        emb = emb.swapdims(0, 1)
        emb = self.final_dense(emb)

        if self.include_cls:
            cls_emb = emb.narrow(1, 0, 1).squeeze(dim=1)
            entity_emb = emb.narrow(1, 1, emb.shape[1] - 1)
            return cls_emb, entity_emb
        else:
            return emb

    def forward(self, *entities: torch.Tensor, mask=None):
        if len(entities) == 1:
            return self.forward_entities(entities[0], mask=mask)
        else:
            balls, boosts, players = entities
            return self.forward_categories(balls, boosts, players, mask=mask)

    def __repr__(self):
        return f"EARL(n_dims={self.n_dims},n_layers={self.n_layers},n_heads={self.n_heads})"
