import torch
from torch import nn as nn

from .earl import EARL


class DotProductPrediction(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim

        self.linear1 = nn.Linear(input_dim, emb_dim)
        self.linear2 = nn.Linear(input_dim, emb_dim)

    def forward(self, inp1, inp2):
        bs = inp1.shape[0]
        emb1 = self.linear1(inp1)
        emb2 = self.linear2(inp2)
        res = torch.bmm(emb1.view(bs, -1, self.emb_dim), emb2.view(bs, self.emb_dim, -1))
        return res


class EARLReplayModel(nn.Module):
    def __init__(self, earl: EARL = None):
        super().__init__()
        if earl is None:
            self.earl = EARL()
        else:
            self.earl = earl

        self.score = nn.Linear(earl.n_dims, 2)
        self.next_touch = DotProductPrediction(earl.n_dims, earl.n_dims // earl.n_heads)
        self.boost_collect = DotProductPrediction(earl.n_dims, earl.n_dims // earl.n_heads)
        self.demo = DotProductPrediction(earl.n_dims, earl.n_dims // earl.n_heads)

    def forward(self, *x):
        cls, ball, boosts, blue, orange = self.earl(*x)
        players = torch.cat((blue, orange), dim=-2)
        return self.score(cls), self.next_touch(players, ball), self.boost_collect(boosts, players), self.demo(players,
                                                                                                               players)

    def __repr__(self):
        return f"EARLReplayModel({repr(self.earl)})"
