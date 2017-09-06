import sys
import random
from collections import namedtuple,deque
sys.path.append('..')
from utils import sum_tree

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','done'))

class ReplayMemory(object):
    def __init__(self, capacity,batch_size):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.batch_size = batch_size

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self ):
        if self.position < self.batch_size:
            return None
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)

class PriorReplayMemory(object):
    def __init__(self, memory_size, batch_size, alpha):
        self.tree = sum_tree.SumTree(memory_size)
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.alpha = alpha

    def push(self, *args):
        priority=1
        data=Transition(*args)
        self.tree.add(data, priority**self.alpha)

    def sample(self, beta):
        if self.tree.filled_size() < self.batch_size:
            return None, None, None
        out = []
        indices = []
        weights = []
        priorities = []
        for _ in range(self.batch_size):
            r = random.random()
            data, priority, index = self.tree.find(r)
            priorities.append(priority)
            weights.append((1./self.memory_size/priority)**beta if priority > 1e-16 else 0)
            indices.append(index)
            out.append(data)
            self.priority_update([index], [0]) # To avoid duplicating
            
        self.priority_update(indices, priorities) # Revert priorities
        print(weights)
        weights = [float(i)/max(weights) for i in weights]
        return out, weights, indices

    def priority_update(self, indices, priorities):
        for i, p in zip(indices, priorities):
            self.tree.val_update(i, p**self.alpha)
    
    def reset_alpha(self, alpha):
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [self.tree.get_val(i)**-old_alpha for i in range(self.tree.filled_size())]
        self.priority_update(range(self.tree.filled_size()), priorities)

