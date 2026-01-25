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
        if isinstance(grad, (list, tuple)):
            grad = grad[0]
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


class prox_MM_l1_UR(BaseUR):
    """
    prox_MM_l1_UR
    A proximal-majorization-minimization (MM) update rule implementing L1 (lasso)
    regularization using a diagonal majorant. Designed to be used as an update
    operator in iterative optimization / learned optimization schemes.
    
    Attributes
    ----------
    lambda_l1 : float
        Radius / strength of the L1 regularization (λ). Controls the amount of sparsity enforced by the proximal / projection steps.
    Expected inputs
    ---------------
    z : torch.Tensor
        Current iterate / state tensor.
    grad : torch.Tensor
        Gradient (or gradient-like) tensor with the same shape as z.
    P : torch.Tensor
        Diagonal majorant tensor (same shape as z). All operations assume
        elementwise multiplication with P (i.e., P plays the role of per-element
        scaling).
    step : int
        Current iteration index (may be unused by some strategies).
    
    Primary methods
    ---------------
    forward(x, grad, P, step)
        Compute an MM update step from the current state x and gradient grad.
        The implementation forms the MM-step z = x - grad (or equivalently the
        MM descent direction) and then applies a sparsity-inducing operator.
        Two intended behaviors are:
          - prox_l1(z, P, lambda_l1): elementwise soft-thresholding with
            thresholds = lambda_l1 * P (solves min_x 0.5*(x-z)^T Diag(P) (x-z)
            + lambda ||x||_1).
          - proj_Dl1ball(z, P, lambda_l1): weighted L1-ball projection that solves
            min_y 0.5 * (z-x)^T Diag(P) (z-x)  subject to  ||x||_1 <= lambda_l1.
        The forward method returns a tensor of the same shape and device as z.
    prox_l1(z, P, lambda_l1)
        Static method that applies the proximal operator for L1 regularization
        with a diagonal majorant. Implements elementwise soft-thresholding.
    proj_Dl1ball(z, P, lambda_l1)
        Static method that projects the input tensor onto a weighted L1 ball
        defined by the diagonal majorant P and radius lambda_l1.

    Returns
    -------
    All methods return torch.Tensor objects with the same shape and device as
    their input state tensors.
    """
    def __init__(self, lambda_l1=0.01, mode="proj_Dl1ball"):
        super().__init__() 
        self.lambda_l1 = lambda_l1
        self.mode = mode # "prox_l1" or "proj_Dl1ball"

    def forward(self, x, grad, step):
        """
        Args:
            x    : current state (Tensor)
            grad : tuple (grad, P) from grad_mod
            P    : diagonal majorant matrix (Tensor, same shape as x)
            step : int, step index
        """
        grad, P = grad  # grad is a tuple (grad, P)
        MM_step = x - grad
        if self.mode == "prox_l1":
            return self.prox_l1(MM_step, P, self.lambda_l1)
        elif self.mode == "proj_Dl1ball":
            return self.proj_Dl1ball(MM_step, P, self.lambda_l1)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def prox_l1(self, z, P, lambda_l1):
        """
        Executes the Proximal MM step for L1 Regularization (Lasso) using a diagonal majorant.

        This function solves the sub-problem:
        min_x  0.5 * (x - z)^T * P * (x - z) + lambda * ||x||_1
        where P is diagonal majorant.

        Args:
            z (torch.Tensor): The MM step direction (z = x_k - P * grad).
            P (torch.Tensor): The diagonal majorant matrix.
            lambda_l1 (float): The regularization strength.

        Returns:
            torch.Tensor: The updated x_{k+1} after applying the proximal operator.
        """
        
        # Compute Thresholds
        thresholds = lambda_l1 * P

        # Apply Soft Thresholding: S(z, t) = sign(z) * max(|z| - t, 0)
        sign_z = torch.sign(z)
        magnitude = torch.clamp(torch.abs(z) - thresholds, min=0.0)
        return sign_z * magnitude

    def proj_Dl1ball (self, z, P, lambda_l1):
        """
        Projection onto the weighted l1 ball defined by:
            C = {z : ||z||_1 <= lambda_l1} with weights P (diagonal of majorant)

        Solves the optimization problem:
            min_{y in C} 1/2 ||z - y||_Diag(P)^2

        Args:
            z (torch.Tensor): Input tensor to be projected.
            P (torch.Tensor): Weights for the l1 norm (diagonal of majorant).
            lambda_l1 (float): Radius of the l1 ball.

        Returns:
            torch.Tensor: Projected tensor onto the weighted l1 ball.
        """
        # Flatten tensors for indexed sorting
        original_shape = z.shape
        z_flat = z.view(-1)
        P_flat = P.view(-1)

        if torch.norm(z, p=1) <= lambda_l1:
            return z

        u, idx = torch.sort(torch.abs(z_flat) / P_flat, descending=True)
        v = P_flat[idx]
        z = torch.abs(z_flat[idx])

        cssv = (torch.cumsum(z, dim=0) - lambda_l1)/torch.cumsum(v, dim=0)

        ind = torch.arange(1, len(z)+1, device=z.device)
        cond = u - cssv > 0
        
        rho_indices = torch.nonzero(cond, as_tuple=False)
        if rho_indices.size(0) > 0:
            rho = rho_indices[-1]
            tau = cssv[rho]
        else:
            tau = 0.0 # Fallback

        return (torch.sign(z) * torch.clamp(torch.abs(z) - tau*P_flat, min=0.0)).view(original_shape)