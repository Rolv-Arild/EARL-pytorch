import torch  # will use pyTorch to handle NN
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from random import sample

# Visualization
import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np
from torch.distributions import Categorical

from earl_pytorch import EARL
from earl_pytorch.model import EARLActorCritic


def save_checkpoint(model, filename):
    torch.save(model.state_dict(), filename)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, player_index, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10, layer_size=128, std=0.1, max_size_buffer=8000):
        self.player_index = player_index
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.std = std

        self.actor_critic = EARLActorCritic(EARL(256, 128, 128))
        self.memory = ReplayBuffer(batch_size, max_size_buffer)
        self.step = 0
        self.intervals = []
        self.evals = []
        self.best = -float("inf")

    def store(self, state, action, probs, vals, reward, done, new_state):
        self.memory.insert(state, action, probs, vals, reward, done, new_state)

    def save_checkpoint(self, model, filename):
        torch.save(model.state_dict(), filename)

    def save(self):
        raise NotImplementedError

    def save_best(self):
        raise NotImplementedError

    def load_best(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def choose_action(self, state):
        state = [torch.tensor(v, dtype=torch.float).to(device) for v in state]

        value, *actions = self.actor_critic(*state)

        dists = [Categorical(logits=act[:, self.player_index]) for act in actions]
        action = np.concatenate([
                dist.sample().swapdims(0, 1).detach().cpu().numpy()
                for dist in dists],
                axis=1)
        probs = [dist.log_prob(action).detach().cpu() for dist in dists]
        value = value.detach().cpu().numpy()

        return action[0], probs[0], value[0]

    def learn(self):

        for e in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, next_states_arr = self.memory.create_batch()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr) - 1):
                # discount = 1
                # a_t = 0
                # for k in range(t, len(reward_arr)-1):
                #     a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                #             (1-int(dones_arr[k])) - values[k])
                #     discount *= self.gamma*self.gae_lambda
                # advantage[t] = a_t

                advantage[t] = reward_arr[t] + self.gamma * (
                        self.actor_critic(next_states_arr[t])[0] - self.actor_critic(state_arr[t])[1])
            advantage = torch.tensor(advantage).to(device)

            values = torch.tensor(values).to(device)

            states = torch.tensor(state_arr, dtype=torch.float).to(device)
            old_probs = torch.tensor(old_prob_arr).to(device)
            actions = torch.tensor(action_arr).to(device)

            critic_value, dist = self.actor_critic(states)

            critic_value = torch.squeeze(critic_value)

            new_probs = dist.log_prob(actions)
            prob_ratio = new_probs.exp() / old_probs.exp()

            batch_a = advantage.unsqueeze(-1)
            weighted_probs = batch_a * prob_ratio
            weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * batch_a
            actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
            returns = advantage + values
            critic_loss = (returns - critic_value) ** 2
            critic_loss = critic_loss.mean()

            total_loss = actor_loss + 0.5 * critic_loss
            self.actor_critic.optimizer.zero_grad()
            total_loss.backward()
            self.actor_critic.optimizer.step()
        if self.memory.is_full():
            self.memory.clear()


class ReplayBuffer(object):
    def __init__(self, batch_size, max_size):
        self.max_size = max_size
        self.batch_size = batch_size
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []

    def insert(self, state, action, probs, vals, reward, done, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)

    def create_batch(self):
        states = []
        actions = []
        probs = []
        vals = []
        rewards = []
        dones = []
        next_states = []
        n_states = len(self.states)
        for i in range(self.batch_size):
            a = random.randint(0, n_states - 1)

            states.append(self.states[a])
            actions.append(self.actions[a])
            probs.append(self.probs[a])
            vals.append(self.vals[a])
            rewards.append(self.rewards[a])
            dones.append(self.dones[a])
            next_states.append(self.next_states[a])

        return np.array(states), np.array(actions), np.array(probs), np.array(vals), np.array(rewards), np.array(
            dones), np.array(next_states)

    def is_full(self):
        return len(self.states) >= self.max_size

    def clear(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError
