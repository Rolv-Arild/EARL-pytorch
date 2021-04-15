import torch
from torch import nn as nn


class EARL(nn.Module):
    N_PARAMETERS = 21

    def __init__(self, n_dims=128, n_layers=4, n_heads=4):
        super().__init__()
        self.n_dims = n_dims
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.emb = nn.Parameter(torch.Tensor(self.N_PARAMETERS), requires_grad=True)
        nn.init.normal_(self.emb)
        self.initial_dense = nn.Linear(self.N_PARAMETERS, n_dims)
        self.final_dense = nn.Linear(n_dims, n_dims)
        encoder_layer = nn.TransformerEncoderLayer(n_dims, n_heads, n_dims)
        encoder_norm = nn.LayerNorm(n_dims)

        self.blocks = nn.TransformerEncoder(
            encoder_layer, n_layers, encoder_norm
        )

    def forward(self, balls, boosts, blue, orange):
        cls = self.emb.repeat(balls.shape[0], 1, 1)
        conc = torch.cat((cls, balls, boosts, blue, orange), dim=1)

        emb = self.initial_dense(conc)
        emb = emb.swapdims(0, 1)
        emb = self.blocks(emb)
        emb = emb.swapdims(0, 1)
        emb = self.final_dense(emb)

        n_balls = balls.shape[1]
        n_boosts = boosts.shape[1]
        n_blue = blue.shape[1]
        n_orange = orange.shape[1]

        cls_emb = emb.narrow(1, 0, 1).squeeze(dim=1)
        entities = emb.narrow(1, 1, emb.shape[1] - 1)
        ball_emb = entities.narrow(1, 0, n_balls)
        boost_emb = entities.narrow(1, n_balls, n_boosts)
        blue_emb = entities.narrow(1, n_balls + n_boosts, n_blue)
        orange_emb = entities.narrow(1, n_balls + n_boosts + n_blue, n_orange)

        return cls_emb, ball_emb, boost_emb, blue_emb, orange_emb
