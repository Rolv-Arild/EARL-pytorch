from typing import Optional

import torch
from torch import nn as nn

from ..util.constants import DEFAULT_FEATURES


class EPRL(nn.Module):  # Experimental alternative to EARL, P stands for pooling-based
    def __init__(self, n_dims: int = 512, n_hidden: int = 4, n_features: Optional[int] = None):
        super().__init__()
        self.n_dims = n_dims
        self.n_features = n_features or len(DEFAULT_FEATURES)

        self.initial_fully_connected = nn.Sequential(
            *([nn.Linear(n_features, n_dims)] + [nn.Linear(n_dims, n_dims) for _ in range(n_hidden - 1)]))

        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.final_fc = nn.Sequential(
            *([nn.Linear(4 * n_dims, n_dims)] + [nn.Linear(n_dims, n_dims) for _ in range(n_hidden - 1)]))

    def forward(self, main_player: torch.Tensor, other_players: torch.Tensor,
                balls: torch.Tensor, boosts: torch.Tensor):
        main_player_emb = self.initial_fully_connected(main_player)
        other_players_emb = self.initial_fully_connected(other_players)
        balls_emb = self.initial_fully_connected(balls)
        boosts_emb = self.initial_fully_connected(boosts)

        main_player_emb = self.max_pool(main_player_emb.swapdims(1, 2)).squeeze(2)
        other_players_emb = self.max_pool(other_players_emb.swapdims(1, 2)).squeeze(2)
        balls_emb = self.max_pool(balls_emb.swapdims(1, 2)).squeeze(2)
        boosts_emb = self.max_pool(boosts_emb.swapdims(1, 2)).squeeze(2)

        emb = torch.cat((main_player_emb, other_players_emb, balls_emb, boosts_emb), dim=-1)

        emb = self.final_fc(emb)

        return emb