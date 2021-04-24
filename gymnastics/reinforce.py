from typing import Any

import numpy as np
import rlgym
import torch
from rlgym.utils import ObsBuilder
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM
from rlgym.utils.gamestates import PlayerData, GameState

from earl_pytorch.dataset.create_dataset import get_base_features, normalize
from earl_pytorch.model import EARLActorCritic
from earl_pytorch.util.util import boost_locations


class EARLObsBuilder(ObsBuilder):
    def __init__(self):
        super().__init__()
        self.last_state = None
        self.last_res = None
        self.first_call_flag = True

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        if self.first_call_flag:
            self.first_call_flag = False
            return np.zeros((1 + 34 + 6, 21))
        if state == self.last_state:
            return self.last_res
        n_blue = n_orange = 0
        for p in state.players:
            if p.team_num == BLUE_TEAM:
                n_blue += 1
            elif p.team_num == ORANGE_TEAM:
                n_orange += 1
        x_ball, x_boost, x_blue, x_orange = get_base_features(1, n_blue, n_orange, include_y=False)

        x_ball[0, :, 4:7] = state.ball.position
        x_ball[0, :, 13:16] = state.ball.linear_velocity
        x_ball[0, :, 16:19] = state.ball.angular_velocity

        for i, (is_active, (x, y, z)) in enumerate(zip(state.boost_pads, boost_locations)):
            x_boost[0, i, 4:7] = x, y, z
            if z > 72:
                x_boost[0, i, 19] = 100.
            else:
                x_boost[0, i, 19] = 12.
            x_boost[0, i, 20] = 1 - is_active

        b = o = 0
        for player in state.players:
            if player.team_num == BLUE_TEAM:
                x_team = x_blue
                i = b
                b += 1
            elif player.team_num == ORANGE_TEAM:
                x_team = x_orange
                i = o
                o += 1
            else:
                print("Invalid team:", player.team_num)
                continue

            x_team[0, i, 4:7] = player.car_data.position
            x_team[0, i, 7:10] = player.car_data.forward()
            x_team[0, i, 10:13] = player.car_data.up()
            x_team[0, i, 13:16] = player.car_data.linear_velocity
            x_team[0, i, 16:19] = player.car_data.angular_velocity
            x_team[0, i, 19] = player.boost_amount
            x_team[0, i, 20] = player.is_alive

        x_data = [x_ball, x_boost, x_blue, x_orange]

        normalize(x_data)

        self.last_state = state
        self.last_res = x_data
        return x_data


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    mdl = torch.load("../out/models/EARLActorCritic_trained.model.ep12")
    mdl.eval()

    env = rlgym.make("DuelSelf", obs_builder=EARLObsBuilder(), game_speed=1, tick_skip=4)
    obs, reward, done, info = env.step(np.zeros((8,)))
    while True:
        pred = mdl(*(torch.from_numpy(v).float() for v in obs[0]))
        actions = EARLActorCritic.convert_actor_result(pred[1], stochastic=True)
        obs, reward, done, info = env.step(actions)
