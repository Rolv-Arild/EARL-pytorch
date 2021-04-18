import torch
from torch import nn as nn

from .earl import EARL


class EARLActorCritic(nn.Module):
    def __init__(self, earl: EARL = None):
        super().__init__()
        if earl is None:
            self.earl = earl
        else:
            self.earl = EARL()

        self.critic = nn.Linear(earl.n_dims, 1)
        self.ground = nn.Linear(earl.n_dims, 72)
        self.aerial = nn.Linear(earl.n_dims, 108)

    def forward(self, *x):
        cls, ball, blue, orange = self.earl(*x)
        players = torch.cat((blue, orange), dim=1)
        return self.critic(cls).squeeze(dim=1), self.ground(players).transpose(1, 2), self.aerial(players).transpose(1,
                                                                                                                     2)
