import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import functional as F
from torch.nn import Parameter
import numpy 

def temp_scheduler(max_epoch, device, decay_portion=0.4, tmin=0.2):
    decay_epoch = int(max_epoch * 0.4)
    t0 = 3.0
    tmin = 0.2
    temp = np.linspace(0, max_epoch, max_epoch )
    t_sched = t0 * np.exp(-temp/decay_epoch ) + tmin
    return torch.Tensor(t_sched).to(device)

def sample_gumbel(shape, device='cpu', eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, device='cpu'):
    y = logits + sample_gumbel(logits.size(), device)
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, device, hard=False):
    y = gumbel_softmax_sample(logits, temperature, device)
    if hard:
        shape = y.size()        
        y_hard = torch.FloatTensor(shape).zero_()
        CG = y#.t()
        #print(y.shape)
        y_hard = y_hard.scatter_(1, CG.argmax(1)[:, None], 1.0)
        
        y = (y_hard - y).detach() + y
    return y

class encoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hard=False, device='cpu'):
        super(encoder, self).__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.hard = hard
        self.reset_parameters()
        self.device = device
        
    def reset_parameters(self):
        self.weight1 = Parameter(torch.rand(self.out_dim, self.in_dim))
    def forward(self, xyz, temp): 
        CG = gumbel_softmax(self.weight1.t(), temp, hard=self.hard, device=self.device).t()
        self.CG = CG/CG.sum(1).unsqueeze(1)
        return torch.matmul(self.CG.expand(xyz.shape[0], self.out_dim, self.in_dim), xyz)
    
class decoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(decoder, self).__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.reset_parameters()
    def reset_parameters(self):
        self.weight = Parameter(torch.rand(self.out_dim, self.in_dim))  
    def forward(self, xyz): 
        weight = self.weight
        return torch.matmul(weight.expand(xyz.shape[0], self.out_dim, self.in_dim), xyz)
