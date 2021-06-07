import rlgym
import time
import ppo
import gym
import numpy as np
import matplotlib.pyplot as plt
from rlgym.utils.reward_functions import CombinedReward, MoveTowardsBallReward, GoalReward

# env = rlgym.make("DuelSelf", reward_fn=rewards.MoveTowardsBallReward(), ep_len_minutes=1)

env = rlgym.make("DuelSelf", reward_fn=GoalReward(), ep_len_minutes=0.5, spawn_opponents=False)
# env = rlgym.make("Duel", reward_fn=rlgym.utils.reward_functions.ShootBallReward(), ep_len_minutes=0.5, spawn_opponents=False)
# env = gym.make("Pendulum-v0")
pend_test_intervals = []
pend_test_vals = []


def plot_results(pend_test_intervals, pend_test_vals):
    pend_testplot, = plt.plot(pend_test_intervals, pend_test_vals, 'b-', label='pend_test')

    plt.xlabel("Steps")
    plt.ylabel("Average Rollout Reward")
    plt.title("Rewards Throughout Training")
    plt.legend(handles=[pend_testplot], loc='lower right')
    plt.savefig("../out/misc/result_plot.png")


def evaluate(n_iterations, pend_test, global_steps):
    pend_test_rewards = []

    for i in range(n_iterations):
        obs = env.reset()
        obs_pend_test = obs

        done = False

        pend_test_ep_reward = [0, 0]

        while not done:
            actions_pend_test = pend_test.choose_action(obs_pend_test)[0]
            # actions_pend_test = np.zeros((2, 8))

            actions = actions_pend_test
            new_obs, reward, done, info = env.step(actions)
            pend_test_ep_reward += reward

            obs_pend_test = new_obs

        pend_test_rewards.append(pend_test_ep_reward)

    pend_test_score = np.mean(pend_test_rewards)

    pend_test_intervals.append(global_steps)
    pend_test_vals.append(pend_test_score)

    pend_test.intervals.append(global_steps)
    pend_test.evals.append(pend_test_score)

    # if pend_test_score > pend_test.best:
    #     pend_test.save_best()
    #     pend_test.best = pend_test_score

    print("Step ", global_steps, "| pend_test Avg Reward: {:.2f} |".format(pend_test_score))
    plot_results(pend_test.intervals, pend_test.evals)


n_actions = env.action_space.shape[0]
input_dims = env.observation_space.shape[0]

gamma = 0.99
alpha = 0.0003
gae_lambda = 0.95
policy_clip = 0.2
batch_size = 32
n_epochs = 10
layer_size = 32
std = 0.25
max_size_buffer = 512

max_train_steps = 5000000
global_steps = 0
N = 250
T = 20
episodes = 0

train = True
load = False

agents = [
    ppo.Agent(i, gamma=gamma, alpha=alpha, gae_lambda=0.95,
              policy_clip=policy_clip, batch_size=batch_size, n_epochs=n_epochs, layer_size=layer_size,
              std=std, max_size_buffer=max_size_buffer)
    for i in range(2 * env._match._team_size)
]

pend_test = ppo.Agent(gamma=gamma, alpha=alpha, gae_lambda=0.95,
                      policy_clip=policy_clip, batch_size=batch_size, n_epochs=n_epochs, layer_size=layer_size,
                      std=std, max_size_buffer=max_size_buffer)
pend_test.step = 0

if load:
    pend_test.load()

else:
    evaluate(3, pend_test, pend_test.step)

if train:
    while pend_test.step < max_train_steps:
        episodes += 1
        obs = env.reset()
        obs_pend_test = obs

        done = False
        steps = 0
        pend_test_ep_reward = 0

        while not done:
            actions_pend_test, probs_pend_test, value_pend_test = pend_test.choose_action(obs_pend_test)

            actions = actions_pend_test
            new_obs, reward, done, state = env.step(actions)

            pend_test_ep_reward += reward

            pend_test.store(obs_pend_test, actions_pend_test, probs_pend_test, value_pend_test, reward, done, new_obs)

            obs_pend_test = new_obs

            pend_test.step += 1

            if pend_test.step % T == 0 and len(pend_test.memory.states) > batch_size:
                pend_test.learn()

            if pend_test.step % N == 0:
                evaluate(5, pend_test, pend_test.step)
                pend_test.save()

env.close()
