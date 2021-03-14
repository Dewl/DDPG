import tensorflow as tf
import os
import sys
import numpy as np
import math

# Hyper Parameters
LAYER1_SIZE = 400
LAYER2_SIZE = 300
LEARNING_RATE = 0.0001
TAU = 0.001

model_dir = os.path.join(os.path.split(
    os.path.realpath(__file__))[0], 'model', 'actor')


class ActorNetwork:
    def __init__(self, sess, state_dim, action_dim):
        self.time_step = 0
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Create actor network
        self.create_network(state_dim, action_dim)

        # Create target actor network

    def create_network(state_dim, action_dim):
        W1 = self.variable([state_dim, LAYER1_SIZE], state_dim)

    def variable(self, shape, f):
        return tf.Variable(tf.random.uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))


def variable(shape, f):
    return tf.Variable(tf.random.uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))

if __name__ == '__main__':
    state_dim = 16
    test = variable([state_dim, LAYER1_SIZE], state_dim)
