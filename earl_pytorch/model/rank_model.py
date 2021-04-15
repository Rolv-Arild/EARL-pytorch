import torch
from torch import nn as nn

from .earl import EARL


class EARLRankModel(nn.Module):
    def __init__(self, earl: EARL):
        super().__init__()
        self.earl = earl

        self.rank = nn.Linear(earl.n_dims, 22)

    def forward(self, *x):
        cls, ball, boosts, blue, orange = self.earl(*x)
        players = torch.cat((blue, orange), dim=-2)
        return self.rank(players)
