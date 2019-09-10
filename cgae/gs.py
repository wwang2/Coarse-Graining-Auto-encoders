import torch
import torch.nn.functional as F
import numpy as np


def temp_scheduler(max_epoch, device, decay_portion=0.4, tmin=0.2):
    decay_epoch = int(max_epoch * 0.4)
    t0 = 3.0
    tmin = 0.2
    temp = np.linspace(0, max_epoch, max_epoch)
    t_sched = t0 * np.exp(-temp / decay_epoch) + tmin
    return torch.Tensor(t_sched).to(device)


def sample_gumbel(shape, dtype=None, device=None, eps=1e-10):
    uniform = torch.rand(shape, dtype=dtype, device=device)
    return -torch.log(-torch.log(uniform + eps) + eps)


def sample_gumbel_softmax(logits, temperature, dtype=None, device=None):
    y = logits + sample_gumbel(logits.size(), dtype=dtype, device=device)
    return torch.nn.functional.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, dtype=None, device=None):
    """Sample a gumbel softmax distribution in dim = -1. Returns gumble, straight-through gumble."""
    y = sample_gumbel_softmax(logits, temperature, dtype=dtype, device=device)
    st_y = torch.zeros(y.shape, dtype=dtype, device=device).scatter_(-1, y.argmax(-1).unsqueeze(-1), 1.0)
    st_y = (st_y - y).detach() + y
    return y, st_y
