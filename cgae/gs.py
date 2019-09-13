import torch
import torch.nn.functional


def sample_gumbel(shape, dtype=None, device=None, eps=1e-10):
    uniform = torch.rand(shape, dtype=dtype, device=device)
    return -torch.log(-torch.log(uniform + eps) + eps)


def sample_gumbel_softmax(logits, temperature, dim=-1, dtype=None, device=None):
    y = logits + sample_gumbel(logits.size(), dtype=dtype, device=device)
    return torch.nn.functional.softmax(y / temperature, dim=dim)


def gumbel_softmax(logits, temperature, dim=-1, dtype=None, device=None):
    """Sample a gumbel softmax distribution in dim = -1. Returns gumble, straight-through gumble."""
    y = sample_gumbel_softmax(logits, temperature, dim=dim, dtype=dtype, device=device)
    st_y = torch.zeros(y.shape, dtype=dtype, device=device).scatter_(dim, y.argmax(dim).unsqueeze(dim), 1.0)
    st_y = (st_y - y).detach() + y
    return y, st_y
