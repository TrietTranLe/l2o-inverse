import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseUR(nn.Module, ABC):
    """
    Abstract update rule for L2O.
    Produces Δx given (x, modified_grad, step).
    Child classes must return:
        new_x
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x, grad, step: int):
        """
        Args:
            x     : current iterate (Tensor)
            grad  : gradient after grad_mod (Tensor)
            step  : int, step index

        Returns:
            new_x : updated x
        """
        pass


class Ori_4DVarNetUR(BaseUR):
    """
    Update rule:
        x_{t+1} = x_t - 1/step * grad_mod(grad) - lr * step/n_steps * grad_ori

    lr behavior:
        - float -> fixed scalar
        - None -> trainable scalar LR
    """

    def __init__(self, lr=None, n_steps=10):
        super().__init__()
        self.n_steps = n_steps

        # Option A: learnable scalar LR
        if lr is None:
            self.lr = nn.Parameter(torch.tensor(0.01))
            print("[4DVarNetUR] Using LEARNABLE scalar LR")

        # Option B: fixed scalar LR
        elif isinstance(lr, (float, int)):
            self.register_buffer("lr", torch.tensor(float(lr)))
            print("[4DVarNetUR] Using FIXED scalar LR:", lr)

        else:
            raise TypeError(f"Invalid LR type: {type(lr)}")

    def forward(self, state, grad, step):
        """
        Args:
            state : current state (Tensor)
            grad  : [output from grad_mod, original gradient] (Tensor, same shape as x)
            step  : int, step index
        """
        grad_mod, grad_ori = grad
        state_update = (
                1 / (step + 1) * grad_mod
                + self.lr * (step + 1) / self.n_steps * grad_ori
        )
        return state - state_update