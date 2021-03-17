import os
import sys
import tensorflow as tf
import numpy as np

class DDPG:
    def __init__(self,
                 env,
                 state_dim,
                 action_dim):
        self.name = 'DDPG'
        self.environment = env
        self.state_dim = state_dim
        self.action_dim = action_dim

    def action(self):

    def observe(self):
