import os
import sys
import numpy as np
from replay_buffer_v2 import ReplayBuffer
from actor_network_v2 import ActorNetwork
from critic_network_v2 import CriticNetwork

GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 64
NOISE = 0.1


class Agent:
    def __init__(self, state_dim, action_n, max_size=100000, env=None):
        self.gamma = GAMMA
        self.tau = TAU
        self.memory = ReplayBuffer(max_size, state_dim, action_n)
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.actor_network = ActorNetwork(state_dim, action_n)
        self.critic_network = CriticNetwork(state_dim, action_n)
        self.target_actor = ActorNetwork(state_dim=state_dim, n_actions=action_n, name='target_actor')
        self.target_critic = CriticNetwork(state_dim=state_dim, n_actions=action_n, name='target_critic')

        self.dummy_Q_target_prediction_input = np.zeros((BATCH_SIZE, 1))
        self.dummy_dones_input = np.zeros((BATCH_SIZE, 1))