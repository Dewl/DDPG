import tensorflow as tf
import os
import sys
import numpy as np
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Concatenate
from tensorflow.keras.models import Model

class ActorNetwork:
    def __init__(self, state_dim, n_actions, fc1_dims=512, fc2_dims=512, fc3_dims=512, name='actor', chkpt_dir='tmp/ddpg'):
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.state_n = state_dim[0]

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, '{}_ddpg.h5'.format(self.model_name))

        self.actor_network = self.build_actor_network()

    def build_actor_network(self):
        state_input = Input(shape=self.state_dim)
        dense = Dense(self.fc1_dims, activation='relu')(state_input)
        dense = Dense(self.fc2_dims, activation='relu')(dense)
        dense = Dense(self.fc3_dims, activation='relu')(dense)
        angular_action = Dense(1, activation='tanh')(dense)
        linear_action = Dense(1, activation='sigmoid')(dense)
        out = Concatenate()[angular_action, linear_action]
        model = Model(inputs=state_input, outs=out)
        model.compile(optimizer='adam', loss=self.ddpg_actor_loss)
        return model

    def ddpg_actor_loss(self, y_true, y_pred):
        return -tf.keras.backend.mean(y_true)

    
