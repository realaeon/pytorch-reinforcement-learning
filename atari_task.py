import gym
import sys
import numpy as np
from atari_wrapper import *


class BasicTask:
    def __init__(self):
        self.normalized_state = True

    def normalize_state(self, state):
        return state

    def reset(self):
        state = self.env.reset()
        if self.normalized_state:
            return self.normalize_state(state)
        return state

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        if self.normalized_state:
            next_state = self.normalize_state(next_state)
        return next_state, np.sign(reward), done, info

    def random_action(self):
        return self.env.action_space.sample()

    
class MountainCar(BasicTask):
    name = 'MountainCar-v0'
    success_threshold = -110

    def __init__(self):
        BasicTask.__init__(self)
        self.env = gym.make(self.name)
        self.env._max_episode_steps = sys.maxsize

class CartPole(BasicTask):
    name = 'CartPole-v0'
    success_threshold = 195

    def __init__(self):
        BasicTask.__init__(self)
        self.env = gym.make(self.name).unwrapped
        self.env._max_episode_steps = sys.maxsize
