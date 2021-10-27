from itertools import product

import numpy as np
import torch
from torch import nn

from earl_pytorch import EARL
from earl_pytorch.util.util import NGPModel
from earl_pytorch.util.constants import MIRROR_TRANSFORM, SWAP_TRANSFORM, NORM_TRANSFORM


def pred(model: nn.Module, x_data: np.ndarray):
    player_masks = np.array(list(product((0, 1), repeat=6)))  # Assumes 3v3
    all_masks = np.concatenate((
        np.zeros((64, 2)), player_masks, np.zeros((64, 34))),  # CLS, ball, players, boosts
        axis=1
    ).repeat(4, axis=0)  # Normal, mirror, swap, swap+mirror

    x_data /= NORM_TRANSFORM

    predictions = []
    model.eval()
    with torch.no_grad():
        for i in range(x_data.shape[0]):
            x_batch = x_data[[i] * (player_masks.shape[0] * 4), ...]  # Same sample repeated

            x_batch[1::4] *= MIRROR_TRANSFORM
            x_batch[2::4] *= SWAP_TRANSFORM
            x_batch[3::4] *= SWAP_TRANSFORM * MIRROR_TRANSFORM

            y_pred = model(torch.from_numpy(x_batch).float(), mask=torch.from_numpy(all_masks))
            y_pred = y_pred.detach().cpu().numpy()[:, 1]
            y_pred = (y_pred[::4] + y_pred[1::4] - y_pred[2::4] - y_pred[3::4]) / 4
            predictions.append(y_pred)

    predictions = np.stack(predictions)

    shapleys = np.zeros((x_data.shape[0], 6))
    for p in range(6):
        idx_masks_with = np.where(player_masks[:, p] == 0)
        idx_masks_without = np.where(player_masks[:, p] == 1)
        shapleys[:, p] = (predictions[:, idx_masks_with] - predictions[:, idx_masks_without]).mean(-1).squeeze(-1)

    return shapleys


if __name__ == '__main__':
    mdl = NGPModel(EARL())
    res = pred(mdl, np.ones((90, 42, 24)))

    print("Hei")
