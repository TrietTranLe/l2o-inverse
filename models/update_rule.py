import math
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
    

class MMFallbackingUR(BaseUR):
    """
    A fallbacking update rule that compares the MM-step and the learned-step and chooses the one with lower estimated upper bound on the loss.
    Update rule:
        x_learned = x - P_grad
        x_GD = x - MM_upper_bound * grad
        If U(x_learned) <= U(x_GD): return x_learned
    """

    def __init__(self, early_stopping_rate: float = None):
        super().__init__()
        self.early_stopping_rate = early_stopping_rate

    def forward(self, x, grad, MM_upper_bound, step):
        """
        Args:
            x    : current state (Tensor)
            grad : output from grad_mod (Tensor, same shape as x)
            step  : int, step index
        """
        if isinstance(grad, (list, tuple)):
            P_grad, P, grad = grad
        
        delta_U = - torch.sum(P* grad.detach()**2) + 1/2 * torch.sum(P**2 * MM_upper_bound.detach() * grad.detach()**2)
        print(delta_U)
        if delta_U <= 0:
            return (x - P_grad), delta_U
        else:
            return (x - MM_upper_bound*grad), delta_U


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
        lambda_l1 = self._get_lambda_l1(x, P, step)
        if self.mode == "prox_l1":
            return self.prox_l1(MM_step, P, lambda_l1)
        elif self.mode == "proj_Dl1ball":
            return self.proj_Dl1ball(MM_step, P, lambda_l1)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _get_lambda_l1(self, x, P, step):
        """ Get lambda_l1 for this step. Can be overridden by child classes for dynamic lambda. """
        return self.lambda_l1

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

    def proj_Dl1ball(self, z, P, lambda_l1):
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


class prox_MM_l1_SFAttention_UR(prox_MM_l1_UR):
    """
    prox_MM_l1_UR with a self-factorized attention network to predict lambda_l1 from physical information (y, L).
    Inherits from prox_MM_l1_UR.
    """
    def __init__(self, dim_hidden=8, mode: str="proj_Dl1ball", init_mode: str ="pinv"):
        super().__init__(mode=mode)
        
        self.init_mode = init_mode  # "pinv", "adjoint", "mne"
        self.dim_hidden = dim_hidden

        # Spatial
        # Input: Average over time -> (B, S, 1)
        self.spatial_net = nn.Sequential(
            nn.Conv1d(1, self.dim_hidden, 1, padding="same"),
            nn.ReLU(),
            # nn.Conv1d(self.dim_hidden, self.dim_hidden, 3, padding="same"),
            nn.Conv1d(self.dim_hidden, self.dim_hidden, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.Conv1d(self.dim_hidden, self.dim_hidden, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.Conv1d(self.dim_hidden, 1, 1, padding="same"),
            nn.Sigmoid()
        )
        
        # Temporal
        # Input: Average over sources -> (B, 1, T)
        self.temporal_net = nn.Sequential(
            nn.Conv1d(1, self.dim_hidden, 1, padding="same"),
            nn.ReLU(),
            #nn.Conv1d(self.dim_hidden, self.dim_hidden, 3, padding="same"),
            nn.Conv1d(self.dim_hidden, self.dim_hidden, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.Conv1d(self.dim_hidden, self.dim_hidden, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.Conv1d(self.dim_hidden, 1, 1, padding="same"),
            nn.Sigmoid()
        )

    def compute_lambda(self, y, L):
        x_init = self._init_x(y, L) # (B, S, T)
        
        # Spatial Map
        x_energy_space = torch.mean(torch.abs(x_init), dim=2, keepdim=True) # (B, S, 1)
        spatial_map = self.spatial_net(x_energy_space.permute(0, 2, 1)).permute(0, 2, 1) # (B, S, 1)
        
        # Temporal Map
        x_energy_time = torch.mean(torch.abs(x_init), dim=1, keepdim=True) # (B, 1, T)
        temporal_map = self.temporal_net(x_energy_time) # (B, 1, T)
        
        # Broadcast multiplication
        # (B, S, 1) * (B, 1, T) -> (B, S, T)
        self.lambda_matrix = spatial_map @ temporal_map

    def _get_lambda_l1(self, x, P, step, eps: float = 1e-8):
        max_x = torch.max(torch.abs(x).flatten(start_dim=1), dim=1)[0]
        return self.lambda_matrix * max_x.unsqueeze(1).unsqueeze(1)  # max(|x|)

    def _init_x(self, y, L):
        if self.init_mode == "pinv":
            return torch.linalg.pinv(L) @ y
        elif self.init_mode == "adjoint":
            return L.T @ y
        elif self.init_mode == "mne":
            SNR = 3.0 # assumed SNR
            LLt = L @ L.t()
            lambda_reg = torch.trace(LLt) / (L.shape[0] * (SNR ** 2))
            inv_term = torch.linalg.inv(LLt + lambda_reg * torch.eye(L.shape[0], device=L.device))
            return L.t() @ inv_term @ y
        else:
            raise ValueError(f"Invalid init mode: {self.init_mode}")


class prox_MM_l1_FGatHisEmb_UR(prox_MM_l1_UR):
    """
    prox_MM_l1_UR with:
        a self-factorized attention network to estimate threshold mask from physical information (y, L)
        a linear network to compute threshold amplitude from statistical values of |z| and P.
    Inherits from prox_MM_l1_UR.
    """
    def __init__(self, dim_hidden_gate=8, dim_hidden_alpha=64, mode: str="proj_Dl1ball", init_mode: str ="pinv"):
        super().__init__(mode=mode)
        
        self.init_mode = init_mode  # "pinv", "adjoint", "mne"
        self.dim_hidden_gate = dim_hidden_gate
        self.dim_hidden_alpha = dim_hidden_alpha
        self.dim_history = 10
        self.alpha_net = nn.Sequential(
            nn.Linear(9 + self.dim_history, self.dim_hidden_alpha),
            nn.ReLU(),
            nn.Linear(self.dim_hidden_alpha, self.dim_hidden_alpha),
            nn.ReLU(),
            nn.Linear(self.dim_hidden_alpha, 1),
            nn.Sigmoid()
        )

        # Spatial
        # Input: Average over time -> (B, S, 1)
        self.spatial_net = nn.Sequential(
            nn.Conv1d(1, self.dim_hidden_gate, 1, padding="same"),
            nn.ReLU(),
            # nn.Conv1d(self.dim_hidden_gate, self.dim_hidden_gate, 3, padding="same"),
            nn.Conv1d(self.dim_hidden_gate, self.dim_hidden_gate, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.Conv1d(self.dim_hidden_gate, self.dim_hidden_gate, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.Conv1d(self.dim_hidden_gate, 1, 1, padding="same"),
            nn.Sigmoid()
        )
        
        # Temporal
        # Input: Average over sources -> (B, 1, T)
        self.temporal_net = nn.Sequential(
            nn.Conv1d(1, self.dim_hidden_gate, 1, padding="same"),
            nn.ReLU(),
            #nn.Conv1d(self.dim_hidden_gate, self.dim_hidden_gate, 3, padding="same"),
            nn.Conv1d(self.dim_hidden_gate, self.dim_hidden_gate, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.Conv1d(self.dim_hidden_gate, self.dim_hidden_gate, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.Conv1d(self.dim_hidden_gate, 1, 1, padding="same"),
            nn.Sigmoid()
        )

    def compute_lambda(self, y, L):
        x_init = self._init_x(y, L) # (B, S, T)
        
        # Spatial Map
        x_energy_space = torch.mean(torch.abs(x_init), dim=2, keepdim=True) # (B, S, 1)
        spatial_map = self.spatial_net(x_energy_space.permute(0, 2, 1)).permute(0, 2, 1) # (B, S, 1)
        
        # Temporal Map
        x_energy_time = torch.mean(torch.abs(x_init), dim=1, keepdim=True) # (B, 1, T)
        temporal_map = self.temporal_net(x_energy_time) # (B, 1, T)
        
        # Broadcast multiplication
        # (B, S, 1) * (B, 1, T) -> (B, S, T)
        self.lambda_mask = spatial_map @ temporal_map

    def _get_lambda_l1(self, x, P, step, eps: float = 1e-8):
        if step == 0:
            self._reset_state(x)

        # Compute features
        x_abs = torch.abs(x)
        x_norm = x_abs/(P + eps)
        features = torch.stack([s for tensor in (x_abs, P, x_norm) for s in self._get_statistic_feature(tensor)]).T  # Shape (batch, 9)

        # Predict alpha
        alpha = self.alpha_net(torch.cat([features, self._history], dim=1))

        # Update history
        self._history = torch.cat([alpha, self._history[:, :-1]], dim=1)

        # Scale to get lambda_l1
        return  alpha.unsqueeze(1) * self.lambda_mask * features[:, 2].unsqueeze(1).unsqueeze(1)  # max(|x|)

    def _init_x(self, y, L):
        if self.init_mode == "pinv":
            return torch.linalg.pinv(L) @ y
        elif self.init_mode == "adjoint":
            return L.T @ y
        elif self.init_mode == "mne":
            SNR = 3.0 # assumed SNR
            LLt = L @ L.t()
            lambda_reg = torch.trace(LLt) / (L.shape[0] * (SNR ** 2))
            inv_term = torch.linalg.inv(LLt + lambda_reg * torch.eye(L.shape[0], device=L.device))
            return L.t() @ inv_term @ y
        else:
            raise ValueError(f"Invalid init mode: {self.init_mode}")
    
    def _get_statistic_feature(self, tensor):
        flat_tensor = tensor.flatten(start_dim=1)
        return [torch.mean(flat_tensor, dim=1), torch.std(flat_tensor, dim=1), torch.max(flat_tensor, dim=1)[0]]

    def _reset_state(self, inp):
        size = [inp.shape[0], self.dim_history]
        self._history = torch.full(size, -1.0, device=inp.device)


class prox_MM_l1_LLambda_UR(prox_MM_l1_UR):
    """
    prox_MM_l1_UR with a trainable lambda_l1.
    Inherits from prox_MM_l1_UR.
    """
    def __init__(self, init_lambda=1e-4, mode="proj_Dl1ball"):
        super().__init__(mode=mode)
        import math
        init_log_w = math.log(init_lambda)
        self.log_w = nn.Parameter(torch.tensor(float(init_log_w)))

    def _get_lambda_l1(self, x, P, step, eps: float = 1e-8):
        return torch.exp(self.log_w)


class prox_MM_l1_Linear_UR(prox_MM_l1_UR):
    """
    prox_MM_l1_UR with a Linear network to predict lambda_l1 at each step.
    Inherits from prox_MM_l1_UR.
    """
    def __init__(self, dim_hidden=64, mode="proj_Dl1ball"):
        super().__init__(mode=mode)
        
        self.dim_hidden = dim_hidden
        # NN to predict alpha
        self.alpha_net = nn.Sequential(
            nn.Linear(9, self.dim_hidden),
            nn.ReLU(),
            nn.Linear(self.dim_hidden, self.dim_hidden),
            nn.ReLU(),
            nn.Linear(self.dim_hidden, 1),
            nn.Sigmoid()
        )

    def _get_lambda_l1(self, x, P, step, eps: float = 1e-8):
        """ Predict lambda_l1 using alpha_net based on features from (x, P) """
        x_abs = torch.abs(x)
        x_norm = x_abs/(P + eps)
        features = torch.stack([s for tensor in (x_abs, P, x_norm) for s in self._get_statistic_feature(tensor)]).T  # Shape (batch, 9)

        # Predict alpha
        alpha = self.alpha_net(features).unsqueeze(1)  # Shape (batch, 1, 1)

        # Scale to get lambda_l1
        return  alpha * features[:, 2].unsqueeze(1).unsqueeze(1)  # max(|x|)

    def _get_statistic_feature(self, tensor):
        flat_tensor = tensor.flatten(start_dim=1)
        return [torch.mean(flat_tensor, dim=1), torch.std(flat_tensor, dim=1), torch.max(flat_tensor, dim=1)[0]]


class prox_MM_net_UR(nn.Module):
    """
    prox_MM_l1_UR with a autoencoder as learned proximal.
    """
    def __init__(self, subgradient_prior_net=None, mode: str = "riemann"):
        super().__init__()
        
        # NN as subgradient prior network
        self.subgradient_prior_net = subgradient_prior_net
        self.subgradient_prior = None
        self.mode = "riemann" # "riemann" or "euclid"

    def forward(self, inner_loss, x, grad, step):
        """
        Args:
            x    : current state (Tensor)
            grad : tuple (grad, P) from grad_mod
            P    : diagonal majorant matrix (Tensor, same shape as x)
            step : int, step index
        """
        P_grad, P, grad = grad  # grad is a tuple (P_grad, P, grad)
        z = self.forward_step(x, P_grad)

        subgradient_prior = self.compute_subgradient_prior(z)
        self.subgradient_prior = subgradient_prior.detach()  # Detach to prevent gradients flowing into the subgradient prior net

        with torch.no_grad():
            #P_grad_norm = torch.linalg.norm(P_grad, dim=(-1, -2)).mean().item()
            P_norm = torch.linalg.norm(P, dim=(-1, -2)).mean().item()
            # z_norm = torch.linalg.norm(z, dim=(-1, -2)).mean().item()
            subgrad_norm = torch.linalg.norm(subgradient_prior, dim=(-1, -2)).mean().item()

            #print("P_grad norm:", P_grad_norm)
            print("P norm:", P_norm)
            # print("z norm:", z_norm)
            print("subgradient_prior norm:", subgrad_norm)

        x_next = self.moreau_forward_step(z, P, subgradient_prior)
        return x_next
    
    def forward_step(self, x, update_step):
        z = x - update_step
        if self.mode == "riemann":
            z = torch.nn.functional.normalize(z, p=2, dim=(-1, -2))
        return z

    def moreau_forward_step(self, z, P, subgradient_prior):
        x = z - P * subgradient_prior
        if self.mode == "riemann":
            return torch.nn.functional.normalize(x, p=2, dim=(-1, -2))
        return x

    def compute_subgradient_prior(self, z):
        if self.mode == "riemann":
            z = torch.nn.functional.normalize(z, p=2, dim=(-1, -2))
        return z - self.subgradient_prior_net(z, self.mode)


class prox_MM_l1_Linear_HisEmb_UR(prox_MM_l1_UR):
    """
    prox_MM_l1_UR with a Linear network to predict lambda_l1 at each step.
    Inherits from prox_MM_l1_UR.
    """
    def __init__(self, dim_hidden=48, mode="proj_Dl1ball", downsamp=None):
        super().__init__(mode=mode)

        self.dim_hidden = dim_hidden
        self.dim_history = 10
        self.alpha_net = nn.Sequential(
            nn.Linear(9 + self.dim_history, self.dim_hidden),
            nn.ReLU(),
            nn.Linear(self.dim_hidden, self.dim_hidden),
            nn.ReLU(),
            nn.Linear(self.dim_hidden, 1),
            nn.Sigmoid()
        )

    def reset_state(self, inp):
        size = [inp.shape[0], self.dim_history]
        self._history = torch.full(size, -1.0, device=inp.device)

    def _forward(self, features):
        features = self.alpha_net(features)
        return features

    def _get_lambda_l1(self, x, P, step, eps: float = 1e-8):
        """ Predict lambda_l1 using alpha_net based on features from (x, P) """
        # Compute features
        if step == 0:
            self.reset_state(x)

        x_abs = torch.abs(x)
        x_norm = x_abs/(P + eps)
        features = torch.stack([s for tensor in (x_abs, P, x_norm) for s in self._get_statistic_feature(tensor)]).T  # Shape (batch, 9)

        # Predict alpha
        alpha = self._forward(torch.cat([features, self._history], dim=1))

        # Update history
        self._history = torch.cat([alpha, self._history[:, :-1]], dim=1)

        # Scale to get lambda_l1
        return alpha.unsqueeze(1) * torch.max(x_abs)  # max(|x|)

    def _get_statistic_feature(self, tensor):
        flat_tensor = tensor.flatten(start_dim=1)
        return [torch.mean(flat_tensor, dim=1), torch.std(flat_tensor, dim=1), torch.max(flat_tensor, dim=1)[0]]


class prox_MM_l1_LinearLSTM_UR(prox_MM_l1_UR):
    """
    prox_MM_l1_UR with a LinearLSTM network to predict lambda_l1 at each step.
    Inherits from prox_MM_l1_UR.
    """
    def __init__(self, dim_hidden=48, mode="proj_Dl1ball", downsamp=None):
        super().__init__(mode=mode)

        self.dim_hidden = dim_hidden
        self._state = []

        self.feature_encoder = nn.Sequential(
            nn.Linear(9, dim_hidden),
            nn.ReLU(),
        )

        self.alpha_decoder = nn.Sequential(
            nn.Linear(dim_hidden, 1),
            nn.Sigmoid()
        )
        self.gates = nn.Linear(2*dim_hidden, 4 * dim_hidden)

    def reset_state(self, inp):
        size = [inp.shape[0], self.dim_hidden]
        self._state = [
            torch.zeros(size, device=inp.device),
            torch.zeros(size, device=inp.device),
        ]

    def _forward(self, features):        
        features = self.alpha_decoder(self._forward_LSTM(self.feature_encoder(features)))
        return features

    def _forward_LSTM(self, features):
        hidden, cell = self._state
        gates = self.gates(torch.cat((features, hidden), 1))
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        in_gate, remember_gate, out_gate = map(
            torch.sigmoid, [in_gate, remember_gate, out_gate]
        )
        cell_gate = torch.tanh(cell_gate)
        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self._state = hidden, cell
        return hidden

    def _get_lambda_l1(self, x, P, step, eps: float = 1e-8):
        """ Predict lambda_l1 using alpha_net based on features from (x, P) """
        # Compute features
        if step == 0:
            self.reset_state(x)

        x_abs = torch.abs(x)
        x_norm = x_abs/(P + eps)
        features = torch.stack([s for tensor in (x_abs, P, x_norm) for s in self._get_statistic_feature(tensor)]).T  # Shape (batch, 9)

        # Predict alpha
        alpha = self._forward(features).unsqueeze(1)  # Shape (batch, 1, 1)

        # Scale to get lambda_l1
        return  alpha * features[:, 2].unsqueeze(1).unsqueeze(1)  # max(|x|)

    def _get_statistic_feature(self, tensor):
        flat_tensor = tensor.flatten(start_dim=1)
        return [torch.mean(flat_tensor, dim=1), torch.std(flat_tensor, dim=1), torch.max(flat_tensor, dim=1)[0]]


class prox_MM_l1_LinearGRU_UR(prox_MM_l1_UR):
    """
    prox_MM_l1_UR with a LinearLSTM network to predict lambda_l1 at each step.
    Inherits from prox_MM_l1_UR.
    """
    def __init__(self, dim_hidden=48, mode="proj_Dl1ball", downsamp=None):
        super().__init__(mode=mode)

        self.dim_hidden = dim_hidden
        self._state = []

        self.feature_encoder = nn.Sequential(
            nn.Linear(9, dim_hidden),
            nn.ReLU(),
        )

        self.alpha_decoder = nn.Sequential(
            nn.Linear(dim_hidden, 1),
            nn.Sigmoid()
        )
        self.gates = nn.Linear(2*dim_hidden, 2*dim_hidden)
        self.new_linear = nn.Linear(2*dim_hidden, dim_hidden)


    def reset_state(self, inp):
        size = [inp.shape[0], self.dim_hidden]
        self._state = torch.zeros(size, device=inp.device)

    def _forward(self, features):        
        features = self.alpha_decoder(self._forward_LSTM(self.feature_encoder(features)))
        return features

    def _forward_LSTM(self, features):
        hidden = self._state

        gates = self.gates(torch.cat((features, hidden), 1))
        update_gate, reset_gate = gates.chunk(2, 1)
        update_gate, reset_gate = map(
            torch.sigmoid, [update_gate, reset_gate]
        )
        new_gate = torch.tanh(self.new_linear(torch.cat((features, reset_gate * hidden), 1)))
        hidden = (1 - update_gate) * hidden + update_gate * new_gate

        self._state = hidden
        return hidden

    def _get_lambda_l1(self, x, P, step, eps: float = 1e-8):
        """ Predict lambda_l1 using alpha_net based on features from (x, P) """
        # Compute features
        if step == 0:
            self.reset_state(x)

        x_abs = torch.abs(x)
        x_norm = x_abs/(P + eps)
        features = torch.stack([s for tensor in (x_abs, P, x_norm) for s in self._get_statistic_feature(tensor)]).T  # Shape (batch, 9)

        # Predict alpha
        alpha = self._forward(features).unsqueeze(1)  # Shape (batch, 1, 1)

        # Scale to get lambda_l1
        return  alpha * features[:, 2].unsqueeze(1).unsqueeze(1)  # max(|x|)

    def _get_statistic_feature(self, tensor):
        flat_tensor = tensor.flatten(start_dim=1)
        return [torch.mean(flat_tensor, dim=1), torch.std(flat_tensor, dim=1), torch.max(flat_tensor, dim=1)[0]]


class S_MM_UR(BaseUR):
    """
    A stochastic-majorization-minimization (MM) update rule.
    """
    def __init__(self, temperature: float = 1.0, snr_target: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.snr_target = snr_target
        self.noise_scheduling_rate = 1.0

    def forward(self, E, x, grad, step, steps):
        """
        Args:
            x    : current state (Tensor)
            grad : tuple (P_grad, P, grad) from grad_mod
            P    : diagonal majorant matrix (Tensor, same shape as x)
            step : int, step index
        """
        P_grad, P, grad = grad  # grad is a tuple (P_grad, P, grad)
        mu = -P_grad  # MM descent direction
        if self.noise_scheduling_rate != 0.0 and self.temperature != 0.0:
            #D = self._compute_D_SNR(P_grad, P, grad, step)
            # D = self._compute_D_trapezoidal(E, x, mu, step)
            D = self._compute_D_trapezoidal_grad(x, mu, P, grad, step)
            if step == 0:
                self.D_hist = D.mean().item()
            else:
                self.D_hist += D.mean().item()
            if step >= steps - 1:
                self.D_hist += D.mean().item()
                print(f'Mean of D: {self.D_hist/step}')

            noise = torch.sqrt(2*self.temperature*D) * torch.randn_like(x)
            # if step == 0:
            #     self.log = {"mu": [mu.item(),], "noise": [noise.item(),]}
            # else:
            #     self.log["mu"].append(mu.item())
            #     self.log["noise"].append(noise.item())

            if step == 0:
                self.snr = 0.0
                self.cosine_sim = 0.0
            else:
                self.snr += torch.norm(mu) / (torch.norm(noise) + 1e-8)
                self.cosine_sim += torch.mean(torch.nn.functional.cosine_similarity(
                                mu.flatten(start_dim=1), 
                                noise.flatten(start_dim=1)))
            if step >= steps - 1:
                print(f"\n[Step: {step}] Average SNR of updates: {self.snr/(step + 1):.6f}")
                print(f"[Step: {step}] Average Cosine Similarity of updates: {self.cosine_sim/(step + 1):.6f}")
            # print(f"[Step: {step}] norm mean: {torch.norm(P_grad):.6f}")
            # print(f"[Step: {step}] norm noise: {torch.norm(noise):.6f}")
            # print(f"[Step: {step}] mean d: {D.mean():.6f}")
            # print(f"[Step: {step}] max d: {D.max():.6f}")
            # print(f"[Step: {step}] mean p: {P.mean():.6f}")
            # print(f"[Step: {step}] max p: {P.max():.6f}")
            
            if step < steps:
                return x + mu + self.noise_scheduling_rate*noise  # Add noise to MM step
            else:
                x_hat = self._denoise_Tweedie(x, D, grad)
                
                with torch.no_grad():
                    var_signal = torch.var(x_hat, dim=1).mean() 
                    var_noise_removed = torch.var(x - x_hat, dim=1).mean()
                    eSNR = 10 * torch.log10((var_signal + 1e-8) / (var_noise_removed + 1e-8))
                    print(f"\n[Step: {step}] Estimated SNR of denoising: {eSNR:.6f}")
                
                return x_hat
        else:
            return x + mu  # Pure MM step without noise

    def _compute_D_SNR(self, P_grad, P, grad, step):
        """ Compute a diagonal matrix D based on P, grad, and step."""
        P_norm = P/(P.mean(dim=tuple(range(1, P.ndim)), keepdim=True))
        if step == 0:
            self.D = P_norm.detach()
            return self.D
        
        alpha = (1.0 / self.temperature) * P * grad.pow(2)
        self.D = P_norm + torch.exp(-alpha) * (self.D - P_norm)
        return torch.mean(P_grad**2)/(2*self.temperature*self.snr_target)*torch.clamp(self.D, min=1e-8)

    def _compute_D_trapezoidal(self, E, x, mu, step, eps=1e-8):
        """ Compute a diagonal matrix D based on P, grad, and step."""
        # with torch.no_grad():
        #     if step == 0:
        #         self.mem_dict = {}
        #         self.mem_dict['E'] = E.item() if torch.is_tensor(E) else E
        #         self.mem_dict['x'] = x.detach().clone()
        #         self.mem_dict['mu'] = mu.detach().clone()
        #         self.mem_dict['D'] = torch.clamp(torch.abs(self.mem_dict['mu'])/(torch.abs(self.temperature*self.mem_dict['x']) + eps), min=eps)
        #     else:
        #         delta_E = E - self.mem_dict['E']
        #         delta_x = x - self.mem_dict['x']
        #         D = (self.mem_dict['D'] + delta_x*self.mem_dict['mu']/(2*self.temperature))*torch.exp(torch.clamp(delta_E / self.temperature, max=50.0)) + delta_x*mu/(2*self.temperature)
        #         self.mem_dict['D'] = torch.nan_to_num(D, nan=1e-8, neginf=1e-8)
        #         self.mem_dict['D'] = torch.clamp(self.mem_dict['D'], min=1e-8)

        #         # Update stored values for next step (backward-looking)
        #         self.mem_dict['E'] = E.item() if torch.is_tensor(E) else E
        #         self.mem_dict['x'].copy_(x)
        #         self.mem_dict['mu'].copy_(mu)
        # return self.mem_dict['D']

        if step == 0:
            self.mem_dict = {}
            self.mem_dict['E'] = E
            self.mem_dict['x'] = x.clone()
            self.mem_dict['mu'] = mu.clone()
            # cov_0 = torch.var(x, unbiased=False).item()
            # self.mem_dict['D'] = torch.clamp(torch.abs(self.mem_dict['mu'])*cov_0/(torch.abs(self.temperature*self.mem_dict['x']) + eps), min=eps)
            self.mem_dict['D'] = torch.full_like(mu, fill_value=1e-8)
        else:
            delta_E = E - self.mem_dict['E']
            delta_x = x - self.mem_dict['x']
            D = (self.mem_dict['D'] + delta_x*self.mem_dict['mu']/(2*self.temperature))*torch.exp(torch.clamp(delta_E / self.temperature, max=50.0)) + delta_x*mu/(2*self.temperature)
            self.mem_dict['D'] = torch.nan_to_num(D, nan=eps, neginf=eps)
            self.mem_dict['D'] = torch.clamp(self.mem_dict['D'], min=eps)

            # Update stored values for next step (backward-looking)
            self.mem_dict['E'] = E
            self.mem_dict['x'] = x.clone()
            self.mem_dict['mu'] = mu.clone()
        return self.mem_dict['D']
    

class S_MM_UR_2(BaseUR):
    """
    A stochastic-majorization-minimization (MM) update rule.
    """
    def __init__(self, temperature: float = 1.0, snr_target=1.0, denoising=False):
        super().__init__()
        self.temperature = temperature
        # self.log_temperature = nn.Parameter(torch.tensor(-7.0))
        
        self.denoising = denoising
        self.noise_scheduling_rate = 1.0

        self.mode = "mu"
        self.eps = 1e-8

    def forward(self, E, x, grad, lambda_max, step, step_num):
        """
        Args:
            x    : current state (Tensor)
            grad : tuple (P_grad, P, grad_E) from grad_mod
            P    : diagonal majorant matrix (Tensor, same shape as x)
            step : int, step index
        """
        # self.temperature = torch.exp(self.log_temperature * torch.log(torch.tensor(10)))
        P_grad, P, grad_E = grad  # grad is a tuple (P_grad, P, grad_E)
        
        mu = -P_grad  # MM descent direction
        if self.noise_scheduling_rate != 0.0 and self.temperature != 0.0:
            # Post-trajectory denoising 
            if step >= step_num and self.denoising:
                return self._denoise_Tweedie(x, grad_E)
            
            if step == 0 or not self.denoising:
                x_clean = x.clone() 
            else:
                print("Tweedie")
                x_clean = self._denoise_Tweedie(x, grad_E)

            D = self._compute_D_trapezoidal(x_clean, E, mu, P, grad_E, lambda_max, step)
            if self.mode == 'mud':
                D, mask = D
            
            noise = torch.sqrt(2*self.temperature*D) * torch.randn_like(x)

            # Logging
            with torch.no_grad():
                if step == 0:
                    self.D_hist = D.mean().item()
                    self.P_hist = P.mean().item()
                    self.snr = 0.0
                    self.cosine_sim = 0.0
                else:
                    self.D_hist += D.mean().item()
                    self.P_hist += P.mean().item()
                    self.snr += torch.norm(mu) / (torch.norm(noise) + 1e-8)
                    self.cosine_sim += torch.mean(torch.nn.functional.cosine_similarity(
                                    mu.flatten(start_dim=1), 
                                    noise.flatten(start_dim=1)))

                if step == step_num - 1:
                    print(f"\nMean of D: {self.D_hist/(step + 1):.6f}")
                    print(f"Mean of P: {self.P_hist/(step + 1):.6f}")
                    print(f"[Step: {step}] Average SNR of updates: {self.snr/(step + 1):.6f}")
                    print(f"[Step: {step}] Average Cosine Similarity of updates: {self.cosine_sim/(step + 1):.6f}")

            if self.mode == 'mud':
                # print(f"Capped elements: {(~mask).sum().item()}")
                # print(f"mask size: {mask.size()}")

                # eps = 1e-12

                # # cap_rate = (~mask).float().mean()
                # drift_retention = (
                #     torch.where(mask, mu, torch.zeros_like(mu)).norm() / (mu.norm() + eps)
                # )

                # drift_energy_retention = (
                #     torch.where(mask, mu, torch.zeros_like(mu)).square().sum()
                #     / (mu.square().sum() + eps)
                # )

                # # print("cap rate:", cap_rate.item())
                # print("drift norm retained:", drift_retention.item())
                # print("drift energy retained:", drift_energy_retention.item())
                return x_clean + torch.where(mask, mu, torch.zeros_like(mu)) + self.noise_scheduling_rate*noise  # Add noise to MM step
            else:
                return x_clean + mu + self.noise_scheduling_rate*noise  # Add noise to MM step
        else:
            return x + mu  # Pure MM step without noise

    def _compute_D_trapezoidal(self, x, E, mu, P, grad, lambda_max, step):
        """ Compute a diagonal matrix D based on P, grad, and step."""
        if step == 0:
            self.mem_dict = {}
            self.mem_dict['x'] = x.clone()
            self.mem_dict['mu'] = mu.clone()
            self.mem_dict['grad'] = grad.clone()
            # self.mem_dict['D'] = self._compute_diffusion_upper_bound(grad, P, lambda_max)
            self.mem_dict['D'] = P.clone()

            # Store initial energy and P for upper bound mode = 'mu'
            self.mem_dict['E'] = E.clone()
            self.mem_dict['E_0'] = E.clone()
            self.mem_dict['P_0'] = P.clone()
            self.mem_dict['W'] = torch.zeros_like(P)  # Initialize W for mode = 'animu'
            # self.mem_dict['D'] = torch.full_like(mu, fill_value=1e-8)
        else:
            delta_x = x - self.mem_dict['x'] # self.mem_dict['mu']
            delta_E = E - self.mem_dict['E'] #(grad + self.mem_dict['grad'])*delta_x/2
            # self.mem_dict['W'] = self.mem_dict['W'] + delta_E

            exponent = torch.clamp(delta_E / self.temperature, max=50.0) # Avoid explosions
            self.mem_dict['D'] = (self.mem_dict['D'] + delta_x*self.mem_dict['mu']/(2*self.temperature))*torch.exp(exponent) + delta_x*mu/(2*self.temperature)
            # self.mem_dict['D'] = torch.nan_to_num(D, nan=eps, neginf=eps)
            
            d_cap = self._compute_diffusion_upper_bound(E, grad, P, lambda_max, eps=self.eps)
                
            if self.mode == "mud":
                keep_mask = self.mem_dict['D'] <= d_cap
                self.mem_dict['D'] = torch.where(keep_mask, self.mem_dict['D'] , d_cap)
            else:
                self.mem_dict['D'] = torch.clamp(self.mem_dict['D'], max=d_cap)

            self.mem_dict['D'] = torch.clamp(self.mem_dict['D'], min=self.eps)

            # Update stored values for next step (backward-looking)
            self.mem_dict['x'] = x.clone()
            self.mem_dict['mu'] = mu.clone()
            self.mem_dict['grad'] = grad.clone()
            self.mem_dict['E'] = E.clone()

        if self.mode == "mud" and step != 0:
            return self.mem_dict['D'], keep_mask
        elif self.mode == "mud" and step == 0:
            return self.mem_dict['D'], torch.ones_like(self.mem_dict['D'], dtype=torch.bool)
        else:
            return self.mem_dict['D']
    
    def _denoise_Tweedie(self, x, grad):
        """ 
        Analytical denoising function based on Tweedie's formula and Stationary Score Matching.
        
        According to Tweedie's formula, the optimal denoised state is:
            x_hat = x + sigma^2 * nabla_x(log p(x))
        
        For a Langevin dynamics system converging to a stationary Gibbs distribution:
            pi(x) = (1/Z) * exp(-E(x) / T) <=> nabla_x(log pi(x)) = -nabla_x(E(x)) / T
            
        Given the injected Langevin noise variance is sigma^2 = 2*T*D, the formula simplifies to:
            x_hat = x + (2*T*D) * (-nabla_x(E(x)) / T) = x - 2 * D * nabla_x(E(x))

        Args:
            x (torch.Tensor): The current noisy state, typically at the final step K.
            D (torch.Tensor): The diffusion matrix (variance modulator) evaluated at x.
            grad (torch.Tensor): The gradient of the energy function nabla_x(E(x)) evaluated at x.

        Returns:
            torch.Tensor: The clean, denoised state x_hat.
        """
        # print(torch.norm((2 * self.mem_dict['D'] * grad).detach()))
        return x - (2 * self.mem_dict['D'] * grad).detach()

    def _compute_diffusion_upper_bound(self, E, grad, P, lambda_max, eps):
        mode = self.mode.lower()
        if mode in ['mix', 'max', 'min', 'geo']:
            lambda_max = torch.clamp(lambda_max, min=eps)
            margin = torch.clamp(1 - lambda_max*P/2.0, min=0.0)
            num = P*margin*(grad**2)
            den = lambda_max*self.temperature
            d_geo = num/den

        if mode == 'mix':
            # d_max = d.amax(dim=list(range(1, d.ndim)), keepdim=True)
            # P_max = P.amax(dim=list(range(1, P.ndim)), keepdim=True)
            # d_tilde = (d/(d_max + eps)) #.detach()
            # P_tilde = P/(P_max + eps)

            # diff_sq = (d_tilde - P_tilde)**2
            # alpha = diff_sq / (d_tilde**2 + P_tilde**2 + eps)

            #alpha = torch.clamp((torch.exp(E/self.temperature) - 1)/(torch.exp(self.mem_dict['E_0']/self.temperature) - 1), min=0.0, max=1.0)
            a = torch.clamp(E / self.temperature, min=0.0)
            b = torch.clamp(self.mem_dict['E_0'] / self.temperature, min=eps)
            a_cap = torch.minimum(a, b)

            num = torch.exp(a_cap - b) * (-torch.expm1(-a_cap))
            den = -torch.expm1(-b)
            alpha = torch.clamp(num / den, min=0.0, max=1.0)

            d_mu = self.mem_dict['P_0'] * torch.exp(torch.clamp((E - self.mem_dict['E_0']) / self.temperature, max=50.0))

            return alpha*d_mu + (1 - alpha)*torch.min(d_mu, d_geo)
        
        elif mode == 'max':
            return torch.max(P, d_geo)
        elif mode == 'min':
            return torch.min(P, d_geo)
        elif mode == 'mu' or mode == 'mud':
            return self.mem_dict['P_0'] * torch.exp(torch.clamp((E - self.mem_dict['E_0']) / self.temperature, max=50.0))
        elif mode == 'animu':
            return self.mem_dict['P_0'] * torch.exp(torch.clamp(self.mem_dict['W'] / self.temperature, max=50.0))
        elif mode == 'geo':
            return d_geo
        elif mode == 'p':
            return P
        else:
            raise ValueError(f"Invalid mode: {mode}")
        

class S_MM_UR_3(BaseUR):
    """
    A stochastic-majorization-minimization (MM) update rule.
    """
    def __init__(self, temperature: float = 1.0, eps=1e-8, D_mode="Const"):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.D_mode = D_mode

    def forward(self, E, x, grad, lambda_max, step, step_num):
        """
        Args:
            x        : current state (Tensor)
            grad     : tuple (P_grad, P, grad_E) from grad_mod
            P        : diagonal majorant matrix (Tensor, same shape as x)
            step     : int, step index
            step_num : int, total number of steps
        """
        P_grad, P, grad_E = grad  # grad is a tuple (P_grad, P, grad_E)
        
        mu = -P_grad  # Descent direction
        if self.temperature != 0.0:
            D = self._compute_D(x, E, mu, P, grad_E, lambda_max, step, step_num)
            noise = torch.sqrt(2*D) * torch.randn_like(x)
            return x + mu + noise
        else:
            return x + mu  # Deterministic step without noise


    def _compute_D(self, x, E, mu, P, grad, lambda_max, step, step_num):
        """ Compute a diagonal matrix D based on P, grad, and step."""
        reduce_dims = tuple(range(1, P.ndim))
        if self.D_mode == "Const":
            if step == 0:
                self.mem_dict = {}
                D0 = self.temperature * P.mean(dim=reduce_dims, keepdim=True)
                self.mem_dict["D0"] = D0
                self.mem_dict["D"] = D0
            return self.mem_dict["D"]

        elif self.D_mode == "CosineAnnealing":
            if step == 0:
                self.mem_dict = {}
                D0 = self.temperature * P.mean(dim=reduce_dims, keepdim=True)
                self.mem_dict["D0"] = D0

            eta_t = 0.5 * (1.0 + math.cos(step / max(step_num - 1, 1) * math.pi))
            D_floor = self.temperature * self.eps
            self.mem_dict["D"] = D_floor + eta_t * (self.mem_dict["D0"] - D_floor)
            return self.mem_dict["D"]

        elif self.D_mode == "Frozen":
            if step == 0:
                self.mem_dict = {}
                self.mem_dict['D'] = self.temperature * P.clone()
            return self.mem_dict['D']

        elif self.D_mode == "Preconditioner":
            return self.temperature * P.clone()

        elif self.D_mode == "IsotropicGGD" or self.D_mode == "GGD":
            D = self._compute_D_trapezoidal(x, E, mu, P, grad, lambda_max, step)
            if self.D_mode == "IsotropicGGD":
                return D.mean(dim=reduce_dims, keepdim=True)
            else:
                return D
        else:
            raise ValueError(f"Invalid D_mode: {self.D_mode}")

    def _compute_D_trapezoidal(self, x, E, mu, P, grad, lambda_max, step):
        """ Compute a diagonal matrix D based on P, grad, and step."""
        if step == 0:
            self.mem_dict = {}
            self.mem_dict['x'] = x.clone()
            self.mem_dict['mu'] = mu.clone()
            self.mem_dict['D'] = self.temperature * P.clone()

            self.mem_dict['E'] = E.clone()
            self.mem_dict['E_0'] = E.clone()
            self.mem_dict['P_0'] = P.clone()
        else:
            delta_x = x - self.mem_dict['x']
            delta_E = E - self.mem_dict['E']

            exponent = torch.clamp(delta_E / self.temperature, max=50.0) # Avoid explosions
            self.mem_dict['D'] = (self.mem_dict['D'] + delta_x*self.mem_dict['mu']/2)*torch.exp(exponent) + delta_x*mu/2
            
            D_cap = self._compute_diffusion_upper_bound(E)
            self.mem_dict['D'] = torch.clamp(self.mem_dict['D'], max=D_cap)

            D_floor = self.temperature * self.eps
            self.mem_dict['D'] = torch.clamp(self.mem_dict['D'], min=D_floor)

            # Update stored values for next step (backward-looking)
            self.mem_dict['x'] = x.clone()
            self.mem_dict['mu'] = mu.clone()
            self.mem_dict['E'] = E.clone()
        return self.mem_dict['D']

    def _compute_diffusion_upper_bound(self, E):
        return self.temperature * self.mem_dict['P_0'] * torch.exp(torch.clamp((E - self.mem_dict['E_0']) / self.temperature, max=50.0))
        