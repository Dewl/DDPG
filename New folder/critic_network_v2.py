import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Concatenate, Input
from tensorflow.keras.models import Model

GAMMA = 0.99
class CriticNetwork:
    def __init__(self, state_dim, n_actions, fc1_dims=512, fc2_dims=512, fc3_dims=512, name='critic', chkpt_dir='tmp/ddpg'):
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.state_n = state_dim[0]
        self.gamma = GAMMA

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, '{}_ddpg.h5'.format(self.model_name))

        self.critic_network = self.build_critic_network()

    def build_critic_network(self):
        state_input = Input(shape=self.state_dim, name='state_input')
        action_input = Input(shape=(self.n_actions,), name='action_input')
        Q_target_prediction_input = Input(shape=(1,), name='Q_target_prediction_input')
        dones_input = Input(shape=(1,), name='dones_input')
        concat_state_action = Concatenate([state_input, action_input])
        dense = Dense(self.fc1_dims, activation='relu')(concat_state_action)
        dense = Dense(self.fc2_dims, activation='relu')(dense)
        dense = Dense(self.fc3_dims, activation='relu')(dense)

        out = Dense(1, activation='linear')(dense)
        model = Model(inputs=[state_input, action_input, Q_target_prediction_input, dones_input], outputs=out)
        model.compile(optimizer='adam', loss=self.ddpg_critic_loss(Q_target_prediction_input, dones_input))
        model.summary()
        return model

    def ddpg_critic_loss(self, Q_target_prediction_input, dones_input):
        def loss(y_true,y_pred):
            target_Q = y_true + (self.GAMMA * Q_target_prediction_input * (1 - dones_input))
            mse = K.losses.mse(target_Q,y_pred)
            return mse
        return loss

# class CriticNetwork(keras.Model):
#     def __init__(self, n_actions, fc1_dims=512, fc2_dims=512, fc3_dims=512, name='critic', chkpt_dir='tmp/ddpg'):
#         super(CriticNetwork, self).__init__()
#         self.fc1_dims = fc1_dims
#         self.fc2_dims = fc2_dims
#         self.fc3_dims = fc3_dims
#         self.n_actions = n_actions

#         self.model_name = name
#         self.checkpoint_dir = chkpt_dir
#         self.checkpoint_file = os.path.join(self.checkpoint_dir, '{}_ddpg.h5'.format(self.model_name))
        
#         self.fc1 = Dense(self.fc1_dims, activation='relu')
#         self.fc2 = Dense(self.fc2_dims, activation='relu')
#         self.fc3 = Dense(self.fc3_dims, activation='relu')
#         self.q = Dense(1, activation=None)

#     def call(self, state, action):
#         action_value = self.fc1(tf.concat([state, action]), axis=1)
#         action_value = self.fc2(action_value)

#         q = self.q(action_value)
#         return q

# class ActorNetwork(keras.Model):
#     def __init__(self, n_actions, fc1_dims=512, fc2_dims=512, fc3_dims=512, name='actor', chkpt_dir='tmp/ddpg'):
#         super(ActorNetwork, self).__init__()
#         self.fc1_dims = fc1_dims
#         self.fc2_dims = fc2_dims
#         self.fc3_dims = fc3_dims
#         self.n_actions = n_actions

#         self.model_name = name
#         self.checkpoint_dir = chkpt_dir
#         self.checkpoint_file = os.path.join(self.checkpoint_dir, '{}_ddpg.h5'.format(self.model_name))

#         self.fc1 = Dense(self.fc1_dims, activation='relu')
#         self.fc2 = Dense(self.fc2_dims, activation='relu')
#         self.fc3 = Dense(self.fc3_dims, activation='relu')
#         self.linear_action = Dense(1, activation='sigmoid')
#         self.angular_action = Dense(1, activation='tanh')
#         self.mu = 
        
#     def call(self, state, action):
#         prob = self.fc1(state)
#         prob = self.fc2(state)
