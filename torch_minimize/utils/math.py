import torch

import sys

def get_f_and_grad(func:"Function", x:torch.Tensor, target_indices:tuple[int])->tuple[torch.float, torch.Tensor]:
    if target_indices is None:
        target_indices = tuple(range(x.shape[0]))

    f = func(x)
    grad_f = torch.autograd.grad(f, x)[0][target_indices,]

    return f.detach().clone(), grad_f.detach().clone()