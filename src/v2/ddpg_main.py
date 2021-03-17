import os
import sys
import rospy
import gym
import gym_gazebo
import numpy as np
import tensorflow as tf

from ddpg import DDPG
from environment import Env

EXPLORATION_DECAY_STEP = 50000
MAXIMUM_STEP_PER_EPOCH = 500

state_dim = 16
action_dim = 2
action_linear_max = 0.25
action_angular_max = 0.5
is_training = False

def main():
    rospy.init_node('ddpg_main')
    env = Env(is_training)
    agent = DDPG(env, state_dim, action_dim)
    past_action = np.array([0., 0.])

    if is_training:
        print('Training mode')
        avg_reward_his = []
        total_reward = 0

        while True:
            state = env.reset()
            one_round_step = 0
            var = 1.

            while True:
                a = agent.action(state)
                a[0] = np.clip(np.random.normal(a[0], var), 0., 1.)
                a[1] = np.clip(np.random.normal(a[1], var), -0.5, 0.5)

                state_, r, done, arrive = env.step(a, past_action)

                ### Store environment trasaction, also store networks' weights here
                time_step = agent.observe(state, a, r, state_, done)

                if arrive:
                    result = 'Success'
                else:
                    result = 'Fail'

                if time_step > 0:
                    total_reward += r

                if time_step % 10000 == 0 and time_step > 0:
                    print('-' * 30)
                    avg_reward = total_reward / 10000
                    print('avg_reward: {}'.format(avg_reward))
                    avg_reward_his.append(avg_reward)
                    print('avg_reward_his: {}'.format(avg_reward_his))
                    total_reward = 0

                if time_step % 5 == 0 and time_step > EXPLORATION_DECAY_STEP:
                    var *= 0.9999

                past_action = a
                state = state_
                one_round_step += 1

                if arrive or done or time_step >= MAXIMUM_STEP_PER_EPOCH:
                    print('Step: {} | Var: {:.2f} | Time_step: {} | {}'.format(one_round_step, var, time_step, result))
                    break

    else:
        print('Testing mode')
        while True:
            state = env.reset()
            one_round_step = 0
            while True:
                a = agent.action(state)
                a[0] = np.clip(a[0], 0., 1.)
                a[1] = np.clip(a[1], -0.5, 0.5)
                state_, r, done, arrive = env.step(a, past_action)
                past_action = a
                state = state_
                one_round_step += 1

                if arrive:
                    print('{} | Arrive !!!'.format(one_round_step))
                    one_round_step = 0

                if done:
                    print('{} | Collision !!!'.format(one_round_step))
                    break

if __name__ == '__main__':
    main()
