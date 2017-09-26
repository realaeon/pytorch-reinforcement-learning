import math
import random
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.multiprocessing as mp
from itertools import count
import numpy as np
import os
from pdb import set_trace 

import sys
sys.path.append('..')
from env.atari_task import Pendulum,MountainCarContinuous,BipedalWalker
from env.env_temp import create_atari_env
from core.model_a3c import ActorCritic
from utils.tools import *
from utils import plot

        
class A3CAgent:
    def __init__(self):
        self.steps_done = 0
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.99
        self.num_steps = 20
        self.max_episode_length=2000
        self.env = create_atari_env('PongDeterministic-v4')
        self.model = ActorCritic(self.env.observation_space.shape[0], self.env.action_space)
        #self.shared_model = shared_model
        #self.optimizer = torch.optim.Adam(self.shared_model.parameters(), lr=1e-4)

    def ensure_shared_grads(self,model, shared_model):
        for param, shared_param in zip(model.parameters(),
                                       shared_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad
        
        
    def train(self,rank,shared_model,optimizer):
        torch.manual_seed(rank)
        self.env.seed(rank)
        self.model.train()

        state = self.env.reset()
        state = torch.from_numpy(state)
        done = True
        reward_=0
        episode_length = 0
        while True:
            # Sync with the shared model
            self.model.load_state_dict(shared_model.state_dict())
            if done:
                cx = Variable(torch.zeros(1, 256))
                hx = Variable(torch.zeros(1, 256))
            else:
                cx = Variable(cx.data)
                hx = Variable(hx.data)

            values = []
            log_probs = []
            rewards = []
            entropies = []

            
            for step in range(self.num_steps):
                episode_length += 1
                value, logit, (hx, cx) = self.model((Variable(state.unsqueeze(0)),
                                                (hx, cx)))
                prob = F.softmax(logit)
                log_prob = F.log_softmax(logit)
                entropy = -(log_prob * prob).sum(1)
                entropies.append(entropy)

                action = prob.multinomial().data
                log_prob = log_prob.gather(1, Variable(action))

                state, reward, done, _ = self.env.step(action.numpy())
                reward_ += reward
                #self.env.render()
                done = done or episode_length >= self.max_episode_length
                reward = max(min(reward, 1), -1)

                if done:
                    episode_length = 0
                    print(os.getpid(),reward_)
                    reward_=0
                    state = self.env.reset()

                state = torch.from_numpy(state)
                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)

                if done:
                    break

            R = torch.zeros(1, 1)
            if not done:
                value, _, _ = self.model((Variable(state.unsqueeze(0)), (hx, cx)))
                R = value.data

            values.append(Variable(R))
            policy_loss = 0
            value_loss = 0
            R = Variable(R)
            gae = torch.zeros(1, 1)
            for i in reversed(range(len(rewards))):
                R = self.gamma * R + rewards[i]
                advantage = R - values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                # Generalized Advantage Estimataion
                delta_t = rewards[i] + self.gamma * \
                    values[i + 1].data - values[i].data
                gae = gae * self.gamma * self.tau + delta_t

                policy_loss = policy_loss - \
                    log_probs[i] * Variable(gae) - 0.01 * entropies[i]

            optimizer.zero_grad()

            (policy_loss + 0.5 * value_loss).backward()
            #print(os.getpid(),policy_loss + 0.5 * value_loss)
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 40)

            self.ensure_shared_grads(self.model, shared_model)
            optimizer.step()

    def run(self,shared_model,optimizer):
        processes = []
        '''
        p = mp.Process(target=test, args=(args.num_processes, args, shared_model))
        p.start()
        processes.append(p)
        '''
        for rank in range(0,4):
            #set_trace()
            p = mp.Process(target=self.train, args=(rank,shared_model,optimizer))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        
if __name__ == '__main__':

    # uncomment when it's fixed in pytorch
    # torch.manual_seed(args.seed)
    env = create_atari_env('PongDeterministic-v4')

    shared_model = ActorCritic(
        env.observation_space.shape[0], env.action_space)
    shared_model.share_memory()
    optimizer =torch.optim.Adam(shared_model.parameters(), lr=1e-4)

    a=A3CAgent()
    a.run(shared_model,optimizer)


