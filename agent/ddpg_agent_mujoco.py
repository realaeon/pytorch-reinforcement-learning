import math
import random
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from itertools import count
import numpy as np
from pdb import set_trace 

import sys
sys.path.append('..')
from env.atari_task import Pendulum,MountainCarContinuous,BipedalWalker,Walker2d
from core.memory import ReplayMemory,Transition
from core.model_ac import ActorNet,CriticNet
from core.policy_greedy import GreedyPolicy
from core.random_process import OrnsteinUhlenbeckProcess
#from tf_logger import Logger
from utils.tools import *
from utils import plot


class DQNAgent:
    def __init__(self):
        self.steps_done = 0
        self.batch_size = 64
        self.GAMMA= 0.99
        self.num_episodes = 2000
        self.task = Walker2d()
        self.memory = ReplayMemory(10000,self.batch_size)
        self.eval_actornet = ActorNet(17,6)
        self.target_actornet = ActorNet(17,6)
        self.target_actornet.load_state_dict(self.eval_actornet.state_dict())
        self.eval_criticnet =  CriticNet(17,6)
        self.target_criticnet = CriticNet(17,6)
        self.target_criticnet.load_state_dict(self.eval_criticnet.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.eval_actornet.parameters(), lr=1e-6)
        self.critic_optimizer = torch.optim.Adam(self.eval_criticnet.parameters(), lr=1e-5,weight_decay=0.01)
        self.random_process = OrnsteinUhlenbeckProcess(size=6,theta=0.15,sigma=0.2)
        self.epsilon = 1
        self.epsilon_d = 1/60000
        self.criterion = torch.nn.MSELoss()
        #self.policy = GreedyPolicy(EPS_START = 0.9,EPS_END = 0.05,EPS_DECAY = 200)
        self.episode_durations= []


    def select_action(self,state):
        action = to_numpy(self.eval_actornet(Variable(state).type(torch.FloatTensor)))
        
        #if self.steps_done % 500 == 0:
            #print(action,to_numpy(self.eval_criticnet(Variable(state).type(torch.FloatTensor),to_tensor(action))),max(self.epsilon,0) * self.random_process.sample())
        if self.steps_done < 1000:
            action = action * 0

        action += max(self.epsilon,0) * self.random_process.sample()
        self.epsilon -= self.epsilon_d
        self.steps_done += 1
        #set_trace()

        return action

    def optimize_model(self):
        transitions = self.memory.sample()
        if not transitions:
            return

        batch = Transition(*zip(*transitions))
        state_batch = Variable(torch.cat(batch.state)).type(torch.FloatTensor)
        action_batch = Variable(torch.cat(batch.action)).type(torch.FloatTensor)
        reward_batch = Variable(torch.cat(batch.reward)).type(torch.FloatTensor)
        next_state_batch = Variable(torch.cat(batch.next_state)).type(torch.FloatTensor)
        done = Variable(torch.cat(batch.done)).type(torch.FloatTensor)
        ##set_trace()
        q_next=self.target_criticnet(next_state_batch,self.target_actornet(next_state_batch))
        q_next = self.GAMMA * q_next * (1 - done)
        q_next.add_(reward_batch)
        q_next=Variable(q_next.data)
        q_ = self.eval_criticnet(state_batch,action_batch)
        if self.steps_done%500 == 0:
            print(q_[0], q_next[0])
        critic_loss = self.criterion(q_, q_next)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)
        self.critic_optimizer.step()
        
        actor_loss = -self.eval_criticnet(state_batch, self.eval_actornet(state_batch)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        #set_trace()
        #if self.steps_done % 200 == 0:
        soft_target_model_updates(self.target_actornet,self.eval_actornet,0.001)
        soft_target_model_updates(self.target_criticnet,self.eval_criticnet,0.001)

    def episode(self):
        episode_num=[]
        episode_durations=[]
        episode_temp=[]
        self.random_process.reset_states()
        for i_episode in range(self.num_episodes):
            state = torch.from_numpy(self.task.reset()).unsqueeze(0)
            for t in count():
                #set_trace()
                #self.task.env.render()
                action = self.select_action(state)#.unsqueeze(0))
                next_state, reward, done, _ = self.task.step(action[0])
                
                #if done:
                    #reward = reward-10
                #reward= - reward * reward
                action = torch.from_numpy(action)#.unsqueeze(0)
                next_state = torch.from_numpy(next_state).unsqueeze(0)
                reward_ = torch.from_numpy(np.array([reward])).unsqueeze(0)
                done_ = torch.from_numpy(np.array([int(done)])).unsqueeze(0)
                self.memory.push(state,action,next_state,reward_, done_)
                state = next_state
                if self.steps_done>3000:
                    self.optimize_model()
                 
                episode_temp.append(reward)

                if (self.steps_done ) % 50 == 0:
                    episode_num.append(self.steps_done + 1)
                    episode_durations.append(episode_temp)
                    episode_temp = []
                    plot.plot_line(episode_num,episode_durations)

                if done  or t > 500:
                    break
                
            if self.steps_done>100000:
                return 1
            
        self.task.env.render(close=True)
        self.task.env.close()

#if __name__ == '__main__':
a= DQNAgent()
a.episode()

