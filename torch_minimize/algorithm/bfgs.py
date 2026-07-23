import torch

from torch_minimize.utils.math import get_f_and_grad


class BFGS:
    def __init__(self, func:"Function", target_indices:tuple[int] = None)->None:
            self.func = func
            self.target_indices = target_indices

    def step(self,
            x_current:torch.Tensor,
            alpha:float,
            x_before:torch.Tensor = None,
            H_current:torch.Tensor = None,
            grad_f_before:torch.Tensor = None,
            grad_f_current:torch.Tensor = None,
            eps:float = 1e-12,
            )->tuple[torch.Tensor, torch.Tensor]:
        target_indices = tuple(range(x_current.shape[0])) if self.target_indices is None else self.target_indices

        if H_current is None:
            H_next = torch.eye(len(target_indices), dtype=x_current.dtype, device=x_current.device)
            if grad_f_current is None:
                _, grad_f_current = get_f_and_grad(func=self.func, x=x_current, target_indices=target_indices)

        else:
            assert H_current.shape == (len(target_indices), len(target_indices)), f"H_current must have shape ({len(target_indices)}, {len(target_indices)}), but got {H_current.shape}"

            if grad_f_current is None:
                _, grad_f_current = get_f_and_grad(func=self.func, x=x_current, target_indices=target_indices)
            if grad_f_before is None:
                _, grad_f_before = get_f_and_grad(func=self.func, x=x_before, target_indices=target_indices)

            with torch.no_grad():
                s = x_current[target_indices,] - x_before[target_indices,]
                y = grad_f_current - grad_f_before

                rho = 1. / (torch.dot(y, s) + eps)
                H_next = (torch.eye(len(target_indices)) - rho * torch.outer(s, y)) @ H_current @ (torch.eye(len(target_indices)) - rho * torch.outer(y, s)) + rho * torch.outer(s, s)

        with torch.no_grad():
            d = -H_next @ grad_f_current

            x_next = x_current.clone()
            x_next[target_indices,] = x_current[target_indices,] + alpha * d

        return x_next.detach().requires_grad_(True), H_next.detach().clone()



class modifiedBFGS:
    def __init__(self, func:"Function", target_indices:tuple[int] = None)->None:
            self.func = func
            self.target_indices = target_indices

    def step(self,
            x_current:torch.Tensor,
            alpha:float,
            x_before:torch.Tensor = None,
            H_current:torch.Tensor = None,
            grad_f_before:torch.Tensor = None,
            grad_f_current:torch.Tensor = None,
            omega:float = 0.2,
            eps:float = 1e-12,
            )->tuple[torch.Tensor, torch.Tensor]:
        target_indices = tuple(range(x_current.shape[0])) if self.target_indices is None else self.target_indices

        if H_current is None:
            H_next = torch.eye(len(target_indices), dtype=x_current.dtype, device=x_current.device)
            if grad_f_current is None:
                _, grad_f_current = get_f_and_grad(func=self.func, x=x_current, target_indices=target_indices)

        else:
            assert H_current.shape == (len(target_indices), len(target_indices)), f"H_current must have shape ({len(target_indices)}, {len(target_indices)}), but got {H_current.shape}"

            if grad_f_current is None:
                _, grad_f_current = get_f_and_grad(func=self.func, x=x_current, target_indices=target_indices)
            if grad_f_before is None:
                _, grad_f_before = get_f_and_grad(func=self.func, x=x_before, target_indices=target_indices)

            with torch.no_grad():
                s = x_current[target_indices,] - x_before[target_indices,]
                y = grad_f_current - grad_f_before

                #####powell damping
                if torch.dot(s, y) < omega * torch.dot(s, H_current @ s):
                    theta = 0.8 * torch.dot(s, H_current @ s) / (torch.dot(s, H_current @ s) - torch.dot(s, y) + eps)
                    y = theta * y + (1 - theta) * H_current @ s
                rho = 1. / (torch.dot(y, s) + eps)
                H_next = (torch.eye(len(target_indices)) - rho * torch.outer(s, y)) @ H_current @ (torch.eye(len(target_indices)) - rho * torch.outer(y, s)) + rho * torch.outer(s, s)

        with torch.no_grad():
            d = -H_next @ grad_f_current

            x_next = x_current.clone()
            x_next[target_indices,] = x_current[target_indices,] + alpha * d

        return x_next.detach().requires_grad_(True), H_next.detach().clone()