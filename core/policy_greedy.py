import numpy as np
import math,random

class GreedyPolicy(object):
    def __init__(self, EPS_START, EPS_END, EPS_DECAY):
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY

    def sample_b(self, steps_done):
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * steps_done / self.EPS_DECAY)
        #print(eps_threshold)
        sample = random.random()
        if sample > eps_threshold:
            return True
        else:
            return False

        





