import os
import sys
import tensorflow as tf
import numpy as np
import pickle
import random
from tensorflow.keras.optimizer import Adam

from replay_buffer import ReplayBuffer
from actor_network import ActorNetwork
from critic_network import CriticNetwork

REPLAY_START_SIZE = 1000

class DDPG:
    def __init__(self,
                 env,
                 state_dim,
                 action_dim,
                 actor_learning_rate=0.0001,
                 critic_learning_rate=0.001,
                 discount_factor=0.99,
                 max_buffer_size=100000):
        self.name = 'DDPG'
        self.environment = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_buffer_size = max_buffer_size
        self.actor_opt = Adam(self.actor_learning_rate)
        self.critic_opt = Adam(self.critic_learning_rate)
        self.action_bound_range = 1
        self.time_step = 0

        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.discount_factor = discount_factor

        self.actor_network = ActorNetwork(state_dim, action_dim).model()
        self.critic_network = CriticNetwork(state_dim, action_dim).model()
        self.target_actor_network = ActorNetwork(state_dim, action_dim).model()
        self.target_actor_network.set_weights(self.actor_network.get_weights())
        self.target_critic_network = CriticNetwork(state_dim, action_dim).model()
        self.critic_network.compile(loss='MSE', optimizer=self.critic_opt)
        self.target_critic_network.set_weights(self.critic_network.get_weights())

        self.replay_buffer = ReplayBuffer(self.max_buffer_size)

    def action(self, state, rand):
        action = self.actor_network.predict(state).ravel()[0]
        if rand:
            return action + random.uniform(-self.action_bound_range, self.action_bound_range)
        else:
            return action

    def train(self):


    def observe(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if self.replay_buffer.count() == self.explore_time:
            print('\n---------------Start training---------------')
        # Store transitions to replay start size then start training
        if self.replay_buffer.count() > self.explore_time:
            self.time_step += 1
            self.train()

        if self.time_step % 10000 == 0 and self.time_step > 0:
            self.actor_network.save('model/actor/actor_model.h5')
            self.critic_network.save('model/critic_model.h5')
            self.actor_network.save_network(self.time_step)
            self.critic_network.save_network(self.time_step)
            with open('buffer', 'wb') as file:
                pickle.dump({'buffer': self.replay_buffer}, file)

        return self.time_step

