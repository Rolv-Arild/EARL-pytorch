import torch
from torch import nn as nn


class NextGoalPredictor(nn.Linear):
    def __init__(self, n_dims, **kwargs):
        super().__init__(n_dims, 2, **kwargs)


class InteractionPredictor(nn.Module):
    def __init__(self, input_dim, emb_dim):
        # Does a single linear layer and dot product between sequences
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim

        self.linear1 = nn.Linear(input_dim, emb_dim)
        self.linear2 = nn.Linear(input_dim, emb_dim)

    def forward(self, inp1, inp2):
        # Returns a tensor of shape (bs, inp1.shape[1], inp2.shape[1])
        # Can be used for things like touch, demo and boost grabs
        # For instance, for predicting next boost one could return logits of (bs, n_players, n_boosts)
        bs = inp1.shape[0]
        emb1 = self.linear1(inp1)
        emb2 = self.linear2(inp2)
        res = torch.bmm(emb1.view(bs, -1, self.emb_dim), emb2.view(bs, self.emb_dim, -1))
        return res


class ControlsPredictorContinuous(nn.Module):
    def __init__(self, n_dims):
        super().__init__()
        self.linear = nn.Linear(n_dims, 8)

    def forward(self, emb):
        actions = self.linear(emb)
        return torch.clip(actions, -1, 1)


class ControlsPredictorDiscrete(nn.Module):
    DEFAULT_SPLITS = (3, 3, 3, 3, 3, 2, 2, 2)

    def __init__(self, n_dims, splits=None):
        super().__init__()
        if splits is None:
            splits = self.DEFAULT_SPLITS
        self.splits = splits
        self.linear = nn.Linear(n_dims, sum(splits))

    def forward(self, emb):
        actions = self.linear(emb)
        return torch.split(actions, self.splits, dim=-1)
