import os
import sys
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras import Model

class CriticNetwork:
    def __init__(self,
                 state_dim,
                 action_dim,
                 fc1_layer=512,
                 fc2_layer=512,
                 fc3_layer=512
                 ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1_layer = fc1_layer
        self.fc2_layer = fc2_layer
        self.fc3_layer = fc3_layer

    def model(self):
        state = Input(shape=self.state_dim, name='stage_input', dtype='float32')
        action = Input(shape=(self.action_dim,), name='action_input')

        x = Concatenate()([state, action])
        x = Dense(self.fc1_layer, activation='relu')(x)
        x = Dense(self.fc2_layer, activation='relu')(x)
        x = Dense(self.fc3_layer, activation='relu')(x)
        out = Dense(1, activation='linear')(x)
        return Model(inputs=[state, action], outs=out)
