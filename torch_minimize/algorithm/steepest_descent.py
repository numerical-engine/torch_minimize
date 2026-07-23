import torch

from torch_minimize.utils.math import get_f_and_grad

class SteepestDescent:
    def __init__(self, func:"Function", target_indices:tuple[int] = None)->None:
        self.func = func
        self.target_indices = target_indices

    def step(self, x:torch.Tensor, alpha:float, grad_f_current:torch.Tensor = None)->torch.Tensor:
        target_indices = tuple(range(x.shape[0])) if self.target_indices is None else self.target_indices
        if grad_f_current is None:
            _, grad_f_current = get_f_and_grad(func=self.func, x=x, target_indices=target_indices)

        x_new = x.clone()
        with torch.no_grad():
            x_new[target_indices,] = x[target_indices,] - alpha * grad_f_current
        return x_new.detach().requires_grad_(True)