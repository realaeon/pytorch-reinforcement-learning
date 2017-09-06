import math
import random
import torch
from torch.autograd import Variable
import torch.nn.functional as F
#import torchvision.transforms as T
import matplotlib.pyplot as plt
from itertools import count

import sys
sys.path.append('..')
from env.atari_task import CartPole
from core.memory import ReplayMemory,Transition
from core.model import Net
#from tf_logger import Logger
from utils import *

class DQNAgent:
    def __init__(self):
        self.steps_done = 0
        self.batch_size = 128
        self.num_episodes = 500
        self.task = CartPole()
        self.memory = ReplayMemory(10000,self.batch_size)
        self.model=Net(4,2)
        self.optimizer = torch.optim.RMSprop(self.model.parameters())
        self.GAMMA= 0.999
        self.episode_durations= [] 

    def select_action(self,state):
        GAMMA = 0.999
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 200

        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            return self.model(
                Variable(state, volatile=True).type(torch.FloatTensor)).data.max(0)[1].view(1, 1)
        else:
            return torch.LongTensor([[random.randrange(2)]])

    def optimize_model(self):
        #if len(self.memory)< self.batch_size:
        #    return
        transitions = self.memory.sample()
        if not transitions:
            return
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,batch.next_state)))
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))

        state_action_values = self.model(state_batch).gather(1, action_batch)
        next_state_values = Variable(torch.zeros(self.batch_size).type(torch.Tensor))
        next_state_values.volatile = False
        ## note GAMMA
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)

#        for param in self.model.parameters():
#            param.grad.data.clamp_(-1, 1)

        self.optimizer.step() 

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.FloatTensor(self.episode_durations)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        plt.pause(0.001)  # pause a bit so that plots are updated

    def episode(self):
        for i_episode in range(self.num_episodes):
            state = torch.FloatTensor(self.task.reset())
            for t in count():
                state=state.view(self.task.env.observation_space.shape[0])
                action = self.select_action(state)
                next_state, reward, done, _ = self.task.step(action[0,0])
                self.task.env.render()
                next_state = torch.FloatTensor(next_state)
                reward = torch.Tensor([reward])

                state=state.view(1,-1)
                next_state=state.view(1,-1)
                self.memory.push(state, action, next_state, reward, done)
                state = next_state

                self.optimize_model()
                if done:
                    self.episode_durations.append(t + 1)
                    self.plot_durations()
                    break
        self.task.env.render(close=True)
        self.task.env.close()
        plot.ioff()
        plot.show()
        

#if __name__ == '__main__':
a= DQNAgent()
a.episode()

