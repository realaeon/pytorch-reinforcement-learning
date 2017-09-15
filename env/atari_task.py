import gym
import sys
import numpy as np
#from env import atari_wrapper

class BasicTask:
    def __init__(self):
        self.normalized_state = False

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
        self.env = gym.make(self.name).unwrapped
        self.env._max_episode_steps = sys.maxsize

class MountainCarContinuous(BasicTask):
    name = 'MountainCarContinuous-v0'
    success_threshold = -110

    def __init__(self):
        BasicTask.__init__(self)
        self.env = gym.make(self.name).unwrapped
        self.env._max_episode_steps = sys.maxsize
        
class CartPole(BasicTask):
    name = 'CartPole-v0'
    success_threshold = 195

    def __init__(self):
        BasicTask.__init__(self)
        self.env = gym.make(self.name).unwrapped
        self.env._max_episode_steps = sys.maxsize

class BipedalWalker(BasicTask):
    name = 'BipedalWalker-v2'
    success_threshold = 300

    def __init__(self):
        BasicTask.__init__(self)
        self.env = gym.make(self.name)
        self.env._max_episode_steps = sys.maxsize
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]

    def step(self, action):
        action = np.clip(action, -1, 1)
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info

class Walker2d(BasicTask):
    name = 'Walker2d-v1'
    success_threshold = 300

    def __init__(self):
        BasicTask.__init__(self)
        self.env = gym.make(self.name)
        self.env._max_episode_steps = sys.maxsize
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]

    def step(self, action):
        action = np.clip(action, -1, 1)
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info
    

    
class Pendulum(BasicTask):
    name = 'Pendulum-v0'
    success_threshold = -10

    def __init__(self):
        BasicTask.__init__(self)
        self.env = gym.make(self.name).unwrapped
        self.env._max_episode_steps = sys.maxsize
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]

    def step(self, action):
        action = np.clip(action, -2, 2)
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info
