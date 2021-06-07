import multiprocessing
import os
import time
from typing import Tuple, Optional, Any

import gym
import numpy as np
import pandas as pd
import torch
from gym.spaces import MultiDiscrete, Box
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.ppo import MlpPolicy
from torch import nn

import rlgym
from earl_pytorch import EARL
from earl_pytorch.model import EARLActorCritic
from earl_pytorch.model.actor_critic import EARLSingleActorCritic
from gymnastics.reinforce import EARLObsBuilder
from gymnastics.vec_monitor import VecMonitor
from rlgym.utils import RewardFunction, ObsBuilder
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.math import scalar_projection
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.reward_functions.common_rewards import SaveBoostReward, VelocityBallToGoalReward, \
    RewardIfTouchedLast, VelocityPlayerToBallReward, DistancePlayerToBallReward, DistanceBallToGoalReward, \
    RewardIfClosestToBall, EventReward, RewardIfBehindBall, ConstantReward, BallYCoordinateReward
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym.wrappers.vec_env_wrapper import SubprocVecEnvWrapper, VecEnvWrapper

import torch as th


class MDWrapper(SubprocVecEnvWrapper):
    def __init__(self, path_to_epic_rl, num_instances, match_args_func):
        super().__init__(path_to_epic_rl, num_instances, match_args_func)
        self.action_space = MultiDiscrete((3, 3, 3, 3, 3, 2, 2, 2))
        self.observation_space = Box(-np.inf, np.inf, (4 + 34 + 1, 23))

    def step_async(self, actions: np.ndarray) -> None:
        actions = np.copy(actions)
        actions[..., :5] -= 1
        super(MDWrapper, self).step_async(actions)


class EARLExtractor(nn.Module):
    def __init__(self, earl):
        super().__init__()
        # self.earl = earl

        # Save dim, used to create the distributions
        self.latent_dim_pi = earl.n_dims
        self.latent_dim_vf = earl.n_dims

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.shared_net = earl
        # self.value_net = nn.Linear(earl.n_dims, earl.n_dims)
        # self.policy_net = nn.Linear(earl.n_dims, earl.n_dims)

        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        entity_emb = self.shared_net(features)
        player_latent = entity_emb[..., 0, :]
        return player_latent, player_latent


class EARLPolicy(MlpPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.earl_actor_critic = EARLActorCritic(EARL(dropout=0., encoder_norm=True,
                                                      n_parameters=23, include_cls=False))
        # self.earl_actor_critic: EARLActorCritic = th.load(
        #     r"..\out\models\EARLActorCritic(EARL(n_dims=256,n_layers=8,n_heads=8))_trained.model.ep4")
        self.mlp_extractor = EARLExtractor(self.earl_actor_critic.earl)

    def _build(self, lr_schedule: Schedule) -> None:
        super(EARLPolicy, self)._build(lr_schedule)
        # weights = th.cat([v.weight for v in self.earl_actor_critic.actor.values()])
        # biases = th.cat([v.bias for v in self.earl_actor_critic.actor.values()])
        # self.action_net.load_state_dict({"weight": weights, "bias": biases})
        # self.value_net.load_state_dict(
        #     {"weight": self.earl_actor_critic.critic.weight, "bias": self.earl_actor_critic.critic.bias})

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        return obs

class TouchIncrementReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.touch_count = 0

    def reset(self, initial_state: GameState):
        self.touch_count = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.ball_touched:
            if player.car_id == state.last_touch:
                self.touch_count += 1
            else:
                self.touch_count = 1
        return self.touch_count


class RewardScaler(RewardFunction):
    lock = multiprocessing.Lock()
    total = multiprocessing.Value("d", 0., lock=False)
    total_squared = multiprocessing.Value("d", 0., lock=False)
    n = multiprocessing.Value("d", 0., lock=False)

    def __init__(self, reward_func: RewardFunction):
        super().__init__()
        self.reward_func = reward_func

    def reset(self, initial_state: GameState):
        self.reward_func.reset(initial_state)
        # print("Values:", self.total.value, self.total_squared.value, self.n.value)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = self.reward_func.get_reward(player, state, previous_action)

        self.lock.acquire()
        total, total_squared, n = self.total.value, self.total_squared.value, self.n.value
        self.total.value += reward
        self.total_squared.value += reward * reward
        self.n.value += 1
        self.lock.release()

        if n > 1:
            var = (total_squared - (total * total) / n) / (n - 1)
            mean = total / n
            scaled_reward = (reward - mean) / (var ** 0.5)
        else:
            scaled_reward = 0

        return scaled_reward


class LogCombinedReward(CombinedReward):
    def __init__(self, reward_functions: Tuple[RewardFunction, ...],
                 reward_weights: Optional[Tuple[float, ...]] = None):
        super().__init__(reward_functions, reward_weights)
        self.totals = np.zeros(len(reward_functions))
        self.n = 0

    def reset(self, initial_state: GameState) -> None:
        super(LogCombinedReward, self).reset(initial_state)
        if self.n > 0:
            print(*(f"{type(rf).__name__}, {tot / self.n}" for rf, tot in zip(self.reward_functions, self.totals)))

    def get_reward(
            self,
            player: PlayerData,
            state: GameState,
            previous_action: np.ndarray
    ) -> float:
        rewards = [
            func.get_reward(player, state, previous_action)
            for func in self.reward_functions
        ]

        self.totals += np.multiply(self.reward_weights, rewards)
        self.n += 1
        return np.dot(self.reward_weights, rewards)

    def get_final_reward(
            self,
            player: PlayerData,
            state: GameState,
            previous_action: np.ndarray
    ) -> float:
        rewards = [
            func.get_final_reward(player, state, previous_action)
            for func in self.reward_functions
        ]

        self.totals += np.multiply(self.reward_weights, rewards)
        self.n += 1
        return np.dot(self.reward_weights, rewards)


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    def get_args():
        fps = 15
        rew = RewardScaler(LogCombinedReward.from_zipped(
            # (EventReward(goal=0, team_goal=100, concede=-100, touch=1, shot=10, save=10, demo=10), 1),
            # (BallYCoordinateReward(), 0.1 / fps),
            # (RewardIfBehindBall(ConstantReward()), 0.01 / fps),
            (VelocityPlayerToBallReward(use_scalar_projection=False), 0.1 / fps),
            # (SaveBoostReward(), 0.01 / fps),
        ))
        # rew = StandStillReward()
        # rew = RewardScaler(SecretSauce(fps))
        return dict(
            team_size=2,
            tick_skip=120 // fps,
            reward_function=rew,
            self_play=True,
            terminal_conditions=[TimeoutCondition(fps * 5 * 60), GoalScoredCondition()],
            obs_builder=EARLObsBuilder()
        )


    rl_path = r"C:\Program Files\Epic Games\rocketleague\Binaries\Win64\RocketLeague.exe"
    env = VecMonitor(MDWrapper(rl_path, 6, get_args))
    # env = VecEnvWrapper(rlgym.make("DuelSelf"))

    model = PPO(EARLPolicy, env, n_epochs=1, target_kl=0.02 / 1.5, learning_rate=1e-4, ent_coef=0.01, vf_coef=1,
                gamma=0.995, verbose=3, batch_size=128, n_steps=2048, tensorboard_log="../out/logs",
                device="cuda")
    # model = PPO.load("policy", env)
    checkpoint = CheckpointCallback(10_000_000 // env.num_envs + 1,
                                    "policy")  # Only increments once all agents take step
    model.learn(100_000_000, callback=checkpoint)
