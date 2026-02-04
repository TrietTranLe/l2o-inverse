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
        lambda_l1 = self._get_lambda_l1(x, P, step)
        if self.mode == "prox_l1":
            return self.prox_l1(MM_step, P, lambda_l1)
        elif self.mode == "proj_Dl1ball":
            return self.proj_Dl1ball(MM_step, P, lambda_l1)
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
