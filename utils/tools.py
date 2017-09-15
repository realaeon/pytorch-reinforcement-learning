import torch
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


def soft_target_model_updates(target, src, tau):
    for target_param, param in zip(target.parameters(), src.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

def to_tensor(ndar, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(torch.from_numpy(ndar), volatile=volatile, requires_grad=requires_grad).type(dtype)
    
