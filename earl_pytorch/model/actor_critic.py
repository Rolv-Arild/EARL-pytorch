import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Categorical

from .earl import EARL


class EARLActorCritic(nn.Module):
    def __init__(self, earl: EARL = None):
        super().__init__()
        if earl is None:
            self.earl = EARL()
        else:
            self.earl = earl

        self.critic = nn.Linear(earl.n_dims, 1)
        self.actor = nn.ModuleDict(dict(
            throttle=nn.Linear(earl.n_dims, 3),  # 0 for full reverse, 2 for full forward
            steer=nn.Linear(earl.n_dims, 3),  # 0 for full left, 2 for full right
            pitch=nn.Linear(earl.n_dims, 3),  # 0 for nose down, 2 for nose up
            yaw=nn.Linear(earl.n_dims, 3),  # 0 for full left, 2 for full right
            roll=nn.Linear(earl.n_dims, 3),  # 0 for roll left, 2 for roll right
            jump=nn.Linear(earl.n_dims, 2),  # 1 for jump, 0 for not
            boost=nn.Linear(earl.n_dims, 2),  # 1 for boost, 0 for not
            handbrake=nn.Linear(earl.n_dims, 2),  # 1 for handbrake, 0 for not
        ))

    @staticmethod
    def convert_actor_result(actor_result, stochastic=False):
        if stochastic:
            actions = np.concatenate([
                Categorical(logits=act.swapdims(1, 2)).sample().swapdims(0, 1).numpy()
                for act in actor_result.values()],
                axis=1)
        else:
            actions = np.concatenate([
                act.argmax(1).swapdims(0, 1).numpy()
                for act in actor_result.values()],
                axis=1)
        actions[:, :5] -= 1
        return actions

    def forward(self, *x):
        cls, ball, boost, blue, orange = self.earl(*x)
        players = torch.cat((blue, orange), dim=1)
        return self.critic(cls).squeeze(dim=1), {key: module(players).swapdims(1, 2) for key, module in
                                                 self.actor.items()}

    def __repr__(self):
        return f"EARLActorCritic({repr(self.earl)})"
