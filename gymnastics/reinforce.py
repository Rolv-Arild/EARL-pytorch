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
        # if self.first_call_flag:
        #     self.first_call_flag = False
        #     self.last_res = np.zeros((1 + 34 + 6, 23))
        #     return self.last_res  # Make Aech happy
        # if state != self.last_state:
        x_ball, x_boost, x_players = get_base_features(1, len(state.players), include_y=False, n_features=23)

        if player.team_num == ORANGE_TEAM:
            boost_pads = state.inverted_boost_pads
            ball = state.inverted_ball
        else:
            boost_pads = state.boost_pads
            ball = state.ball

        x_ball[0, :, 4:7] = ball.position
        x_ball[0, :, 13:16] = ball.linear_velocity
        x_ball[0, :, 16:19] = ball.angular_velocity

        for i, (is_active, (x, y, z)) in enumerate(zip(boost_pads, boost_locations)):
            x_boost[0, i, 4:7] = x, y, z
            if z > 72:
                x_boost[0, i, 19] = 100.
            else:
                x_boost[0, i, 19] = 12.
            x_boost[0, i, 20] = 1 - is_active

        for i, player2 in enumerate(state.players):
            if player2.team_num == player.team_num:
                x_players[0, i, 2] = 1
            else:
                x_players[0, i, 3] = 1

            if player.team_num == ORANGE_TEAM:
                car_data = player2.inverted_car_data
            else:
                car_data = player2.car_data

            x_players[0, i, 4:7] = car_data.position
            x_players[0, i, 7:10] = car_data.forward()
            x_players[0, i, 10:13] = car_data.up()
            x_players[0, i, 13:16] = car_data.linear_velocity
            x_players[0, i, 16:19] = car_data.angular_velocity
            x_players[0, i, 19] = player2.boost_amount * 100
            x_players[0, i, 20] = player2.is_demoed
            x_players[0, i, 21] = player2.on_ground
            x_players[0, i, 22] = player2.has_flip

        p_index = state.players.index(player)
        x_players[0, [0, p_index], :] = x_players[0, [p_index, 0], :]

        x_data = [x_ball, x_boost, x_players]

        normalize(x_data)

        x_data = np.concatenate(x_data[::-1], axis=1).squeeze(axis=0)

        # Handle rare memory corruption
        x_data = np.nan_to_num(x_data.astype(np.float32))
        x_data[np.abs(x_data) > 10] = 0

        self.last_state = state
        self.last_res = x_data
        return x_data


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    mdl = torch.load("../out/models/EARLActorCritic(EARL(n_dims=256,n_layers=8,n_heads=8))_trained.model.ep7")
    # mdl = torch.load("../out/models/EARLActorCritic_trained.model.ep12")
    mdl.eval()

    env = rlgym.make("StandardSelf", obs_builder=EARLObsBuilder(), game_speed=1, tick_skip=4)
    env.reset()
    obs, reward, done, info = env.step(np.zeros((6, 8)))
    while True:
        pred = mdl(*(torch.from_numpy(v).float() for v in obs[0]))
        print(torch.sigmoid(pred[0]).item())
        actions_by_team = EARLActorCritic.convert_actor_result(pred[1], stochastic=True)
        actions = np.zeros((6, 8))
        i = 0
        n_blue = sum(player.team_num == BLUE_TEAM for player in info["state"].players)
        o = b = 0
        for player in info["state"].players:
            if player.team_num == BLUE_TEAM:
                actions[i, :] = actions_by_team[b, :]
                b += 1
            elif player.team_num == ORANGE_TEAM:
                actions[i, :] = actions_by_team[n_blue + o, :]
                o += 1
            i += 1
        obs, reward, done, info = env.step(actions)
