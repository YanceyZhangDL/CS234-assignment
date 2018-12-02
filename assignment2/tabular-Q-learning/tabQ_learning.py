import numpy as np
import gym
import time
from lake_envs import *

import matplotlib.pyplot as plt

def get_action(env, state, Q, epsilon):
  if np.random.random() < epsilon:
    return env.action_space.sample()
  else:
    return np.argmax(Q[state])

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def learn_Q_QLearning(env, num_episodes=5000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99):
  """Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy.
  Update Q at the end of every episode.

  Parameters
  ----------
  env: gym.core.Environment
    Environment to compute Q function for. Must have nS, nA, and P as
    attributes.
  num_episodes: int 
    Number of episodes of training.
  gamma: float
    Discount factor. Number in range [0, 1)
  learning_rate: float
    Learning rate. Number in range [0, 1)
  e: float
    Epsilon value used in the epsilon-greedy method. 
  decay_rate: float
    Rate at which epsilon falls. Number in range [0, 1)

  Returns
  -------
  np.array
    An array of shape [env.nS x env.nA] representing state, action values
  """
  # https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
  # LEFT = 0
  # DOWN = 1
  # RIGHT = 2
  # UP = 3

  ############################
  # YOUR IMPLEMENTATION HERE #
  ############################

  Q = np.zeros((env.nS, env.nA))
  count = 0
  rewards = []
  while count < num_episodes:
    if (count + 1) % 100 == 0:
      e *= decay_rate
      print('finished {0} episodes'.format(count + 1))
    obs = env.reset()
    done = False
    rew = 0
    while not done:
      action = get_action(env, obs, Q, e)
      obs_next, reward, done, info = env.step(action)
      Q[obs][action] += lr * (reward + gamma * np.max(Q[obs_next]) - Q[obs][action])
      obs = obs_next
      rew += reward
    count += 1
    rewards.append(rew)

  # plot the progress
  fig = plt.figure()
  ax = fig.add_subplot(111)
  # avg_rewards = np.array(rewards).cumsum() / np.arange(1, len(rewards) + 1)
  avg_rewards = moving_average(rewards, 1000)
  ax.plot(np.arange(avg_rewards.shape[0]), avg_rewards)
  plt.savefig('./rewards_vs_epoch.png')
  plt.close()

  print('Q table (zeros correspond to terminating states (i.e. H, G):\n{0}'.format(Q))
  return Q


def render_single_Q(env, Q):
  """Renders Q function once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play Q function on. Must have nS, nA, and P as
      attributes.
    Q: np.array of shape [env.nS x env.nA]
      state-action values.
  """

  episode_reward = 0
  state = env.reset()
  done = False
  while not done:
    # env.render()
    # time.sleep(0.1) # Seconds between frames. Modify as you wish.
    action = np.argmax(Q[state])
    state, reward, done, _ = env.step(action)
    episode_reward += reward
  # print "Episode reward: %f" % episode_reward
  return episode_reward

# Feel free to run your own debug code in main!
def main():
  env = gym.make('Stochastic-4x4-FrozenLake-v0')
  Q = learn_Q_QLearning(env)

  rew = 0
  num_episodes = 1000
  for i in range(num_episodes):
    rew += render_single_Q(env, Q)
  rew /= float(num_episodes)
  print('avg score over {0}: {1}'.format(num_episodes, rew))

if __name__ == '__main__':
    main()
