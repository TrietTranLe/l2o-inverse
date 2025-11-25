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


class GradientDescentUR(BaseUR):
    """
    Gradient Descent L2O update rule:
        x_{t+1} = x_t - grad_mod(grad)

    No learnable parameters.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, grad, step):
        """
        Args:
            x    : current state (Tensor)
            grad : output from grad_mod (Tensor, same shape as x)
            step  : int, step index
        """
        return x - grad


class LRUR(BaseUR):
    """
    Update rule:
        x_{t+1} = x_t - lr(step) * grad_mod(grad)

    lr behavior:
        - float -> fixed scalar
        - list/tuple -> per-step LR schedule
        - None -> trainable scalar LR
    """

    def __init__(self, lr=None):
        super().__init__()

        # Option A: learnable scalar LR
        if lr is None:
            self.lr = nn.Parameter(torch.tensor(0.01))
            self.lr_schedule = None
            print("[LRUpdateRule] Using LEARNABLE scalar LR")

        # Option B: fixed scalar LR
        elif isinstance(lr, (float, int)):
            self.register_buffer("lr", torch.tensor(float(lr)))
            self.lr_schedule = None
            print("[LRUpdateRule] Using FIXED scalar LR:", lr)

        # Option C: per-step LR schedule
        elif isinstance(lr, (list, tuple)):
            # store python list - NOT a tensor
            self.lr = None
            self.lr_schedule = [float(v) for v in lr]
            print("[LRUpdateRule] Using PER-STEP LR schedule:", self.lr_schedule)

        else:
            raise TypeError(f"Invalid LR type: {type(lr)}")

    def get_lr(self, step: int):
        """Return LR for this step."""
        # learnable scalar or fixed scalar
        if self.lr:
            return self.lr

        # per-step schedule
        if step < len(self.lr_schedule):
            return torch.tensor(self.lr_schedule[step], device=self.lr_schedule_device)
        else:
            return torch.tensor(self.lr_schedule[-1], device=self.lr_schedule_device)

    # Set automatically when calling .to()
    @property
    def lr_schedule_device(self):
        if hasattr(self, "lr") and isinstance(self.lr, torch.Tensor):
            return self.lr.device

    def forward(self, x, grad, step):
        """
        Args:
            x    : current state (Tensor)
            grad : output from grad_mod (Tensor, same shape as x)
            step  : int, step index
        """
        lr = self.get_lr(step)
        return x - lr * grad

