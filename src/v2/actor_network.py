import os
import sys
import tensorflow as tf
import numpy as np

from tensorflow.keras.initializers import RandomUniform as RU
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras import Model

class ActorNetwork:
    def __init__(self,
                 state_dim,
                 action_dim,
                 fc1_layer=512,
                 fc2_layer=512,
                 fc3_layer=512):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1_layer = fc1_layer
        self.fc2_layer = fc2_layer
        self.fc3_layer = fc3_layer

    def model(self):
        state = Input(shape=self.state_dim, dtype='float32')
        x = Dense(self.fc1_layer, activation='relu', kernel_initializer=RU(-1/np.sqrt(self.state_dim), 1/np.sqrt(self.state_dim)))(state)
        x = Dense(self.fc2_layer, activation='relu', kernel_initializer=RU(-1/np.sqrt(self.fc1_layer), 1/np.sqrt(self.fc1_layer)))(x)
        x = Dense(self.fc3_layer, activation='relu', kernel_initializer=RU(-1/np.sqrt(self.fc2_layer), 1/np.sqrt(self.fc2_layer)))(x)
        action_linear = Dense(1, activaton='sigmoid')(x)
        action_angular = Dense(1, activation='tanh')(x)
        out = Concatenate()([action_linear, action_angular])
        return Model(inputs=state, outs=out)
