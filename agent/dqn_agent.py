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
from core.policy_greedy import GreedyPolicy
#from tf_logger import Logger
from utils import plot

class DQNAgent:
    def __init__(self):
        self.steps_done = 0
        self.batch_size = 32
        self.num_episodes = 2000
        self.task = CartPole()
        self.memory = ReplayMemory(5000,self.batch_size)
        self.model=Net(4,2)
        self.targetM = Net(4,2)
        self.targetM.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(),0.005)
        self.GAMMA= 0.99
        self.policy = GreedyPolicy(EPS_START = 0.9,EPS_END = 0.02,EPS_DECAY = 500)
        self.episode_durations= []
        self.explorat_staps=50

    def select_action(self,state):
        self.steps_done += 1
        if self.policy.sample_b(self.steps_done):
            return self.model(
                Variable(state)).data.max(1)[1][0]
        else:
            return random.randrange(2)

    def optimize_model(self):
        transitions = self.memory.sample()
        if not transitions:
            return
        batch = Transition(*zip(*transitions))

        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        next_state_batch = Variable(torch.cat(batch.next_state))

        state_action_values = self.model(state_batch).gather(1, action_batch)
        next_state_values = self.targetM(next_state_batch).detach().max(1)[0]
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        
        mseloss = torch.nn.MSELoss()
        loss = mseloss(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)
        self.optimizer.step()
        if self.steps_done % 200 == 0:
            self.targetM.load_state_dict(self.model.state_dict())
            #print('load dict')

    def episode(self):
        episode_num=[]
        episode_durations=[]
        episode_temp=[]
        for i_episode in range(self.num_episodes):
            state = torch.FloatTensor(self.task.env.reset()).unsqueeze(0)

            for t in count():
#                state=state.view(self.task.env.observation_space.shape[0])
                action = self.select_action(state)
                next_state, reward, done, _ = self.task.step(action)
#                print(reward)
                if done:
                    reward=-5
                #self.task.env.render()
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                reward = torch.Tensor([reward])
                action = torch.LongTensor([action]).unsqueeze(0)
                self.memory.push(state, action, next_state, reward, done)
                state = next_state
                #if t % 10 ==0:
                self.optimize_model()
                
                if done  or t > 200:
                    episode_temp.append(t + 1)
                    break

            if (i_episode + 1) % 10 == 0:
                episode_num.append(i_episode+1)
                episode_durations.append(episode_temp)
                episode_temp = []
                plot.plot_line(episode_num,episode_durations)
        #self.task.env.render(close=True)
        self.task.env.close()

#if __name__ == '__main__':
a= DQNAgent()
a.episode()

