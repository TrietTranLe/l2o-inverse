import torch 
from torch import nn
import numpy as np


class EsiGradSolver(nn.Module): 
    def __init__(self, reg_net, grad_mod, update_rule, inner_loss, n_steps: int, bound_estimating_steps: int | None = None, init_type: str = "noise", noise_ampl: float = 1e-3, fwd=None,
                # mne_info=None,
                *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_net = reg_net
        self.grad_mod = grad_mod
        self.update_rule = update_rule
        self.inner_loss = inner_loss
        self.n_steps = n_steps
        self.bound_estimating_steps = bound_estimating_steps

        self.init_type = init_type
        self.noise_ampl = noise_ampl
        self.fwd = fwd
        # self.mne_info = mne_info
        # self.inv_op = None 
        # self.reg = 1/25

        self._grad_norm = None

    def forward(self, x: torch.Tensor, y: torch.Tensor, steps: int | None = None, return_all: bool = False):
        """
        Run the inner optimization loop and return the final state.

        Args:
            x: input tensor for initialization (or ignored depending on init rule)
            y: target tensor (needed for computing inner loss)
            steps: number of unrolled inner steps
            return_all: whether to return list of inner losses

        Returns:
            final_state or (final_state, inner_losses)
        """
        if steps is None:
            steps = self.n_steps

        with torch.set_grad_enabled(True):
            # Initialize state
            state = self.init_state(x, y)
            self.grad_mod.reset_state(x)

            inner_losses: list[torch.Tensor] = []
            full_terms: dict = {}

            # Unrolled inner loop
            for step in range(steps):
                # compute inner loss
                inner_loss, terms, intermediate = self.inner_loss(x=state, y=y, AE=self.reg_net, return_terms=True, return_intermediate=True)
                 
                inner_losses.append(inner_loss.detach())
                for k, v in terms.items():
                    if k not in full_terms.keys():
                        full_terms[k] = [v.detach(),]
                    else:
                        full_terms[k].append(v.detach())
                # gradient wrt state
                grad = torch.autograd.grad(
                    inner_loss,
                    state,
                    create_graph=True,
                )[0]                    

                # track grad norm
                self.last_grad_norm = grad.detach().norm()

                # compute bounds (optional)
                if self.bound_estimating_steps is not None:
                    if self.bound_estimating_steps == 0:
                        from collections import deque
                        flatten_dict = {}
                        queue = deque([(None, intermediate)]) # (parent_key, current_dict)
                        while queue:
                            parent, current = queue.popleft()
                            for k, v in current.items():
                                full_key = f"{parent}.{k}" if parent else k
                                if isinstance(v, dict):
                                    queue.append((full_key, v))
                                else:
                                    flatten_dict[full_key] = v
                        print(flatten_dict.keys())
                        print(1/"r")
                        bounds = self.compute_bounds(intermediate, state, grad)
                    else:
                        bounds = self.estimate_bounds(inner_loss, state, grad, self.bound_estimating_steps)
                else:
                    bounds = None

                # apply grad modifier
                grad = self.grad_mod(state, grad, bounds)

                # apply update rule
                state = self.update_rule(state, grad, step)

                # in eval mode, detach to avoid graph growth
                if not self.training:
                    state = state.detach().requires_grad_(True)

            return (state, inner_losses, full_terms) if return_all else state

    def compute_bound_analytical(self, loss_intermediate, lambda_val: float = 1.0, eps_softrelu: float = 0.1, eps_div: float = 1e-8):
        """
        Compute the upper and lower bounds for p(x) using analytical bounds 
        for the Jacobian (rho) and Hessian (alpha), instead of autograd.

        Mathematical bounds:
            Upper Bound = 1 / (mu1 + lambda * mu2) * 1
            Lower Bound = eps

        Args:
            loss_intermediate (dict): Input data dictionary.
                                    Keys required: 'input.L', 'input.x', 'data.L@x', 'reg.AE(x)'
            lambda_val (float): Regularization coefficient lambda.
            eps_softrelu (float): Epsilon used in SoftReLU activation (for alpha calculation).
            eps_div (float): Small constant to avoid division by zero.
        
        Returns:
            lower_bound (Tensor): Tensor of size x (approx 0).
            upper_bound (Tensor): Tensor of size x containing the upper bound values.
        """
        L = loss_intermediate['input.L']
        x = loss_intermediate['input.x']
        Lx = loss_intermediate['data.L@x']      # Pre-computed L*x
        Phi_x = loss_intermediate['reg.AE(x)']  # Pre-computed Phi(x)
        
        s = Phi_x.numel() # Dimension of the output vector s
        
        # Basic Norms
        x_norm = torch.norm(x)
        beta = torch.norm(Phi_x)      # beta = ||Phi(x)||
        Lx_norm = torch.norm(Lx)
        L_op = torch.norm(L) 

        # Compute RHO and ALPHA analytically
        rho = compute_rho_analytical(ae_layers)
        alpha = compute_alpha_analytical(ae_layers, epsilon=eps_softrelu)
        
        # Safeguards to prevent division by zero
        beta_safe = max(beta, eps_div)
        x_norm_safe = max(x_norm, eps_div)
        Lx_norm_safe = max(Lx_norm, eps_div)
        
        # mu1 = 5 * ||L||^2 / ||Lx||^2
        mu1 = 5.0 * (L_op ** 2) / (Lx_norm_safe ** 2)

        # mu2 = (1/(beta*||x||)) * (5*rho + 1) 
        #       + (2*alpha*sqrt(s))/beta 
        #       + 4 * (rho/beta + 1/||x||)^2
        mu2 = (1.0 / (beta_safe * x_norm_safe)) * (5.0 * rho + 1.0) + \
              (2.0 * alpha * np.sqrt(s)) / beta_safe + \
              4.0 * ((rho / beta_safe) + (1.0 / x_norm_safe)) ** 2

        # Upper bound = 1 / (mu1 + lambda * mu2)
        val_upper = 1.999 / (mu1 + lambda_val * mu2 + eps_div)
        
        # Expand scalar result to tensor matching x's shape
        upper_bound = torch.full_like(x, val_upper.item())
        
        # Lower bound (nu > 0, approximated by eps)
        lower_bound = torch.full_like(x, eps_div)

        return lower_bound, upper_bound


    def estimate_bounds(self, loss, state: torch.Tensor, grad: torch.Tensor | None = None, steps: int = 3, safety: float = 1.0, eps: float = 1e-8):
        """
        Estimate a local Hessian norm bound using Hessian-Vector Products (HVP).

        Args:
            loss: Scalar loss tensor evaluated at 'state' (must keep computation graph).
            state: Tensor with respect to which the Hessian is computed (state.requires_grad == True).
            grad: Optional precomputed gradient_state loss with 'create_graph=True'. If None, it is computed internally.
            steps: Number of power-iteration steps (default: 3).
            safety: Multiplicative safety factor for a conservative bound (default: 1.0 - raw estimate).
            eps: Small constant for numerical stability in vector normalization.

        Returns:
            Scalar float representing a local upper bound on ||Hessian(state)||.
        """
        single_input = (loss.dim() == 0)
        if single_input:
            loss = loss.unsqueeze(0)
            if state.dim() > 1:
                state = state.unsqueeze(0)
            else:
                state = state.view(1, -1)

        b = state.shape[0]
        flat_dim = state[0].numel()

        if grad is None:
            grad = torch.autograd.grad(
                outputs=loss, inputs=state,
                grad_outputs=torch.ones_like(loss),
                create_graph=True, retain_graph=True
            )[0]

        grads_flat = grad.view(b, -1)

        v = torch.randn_like(grads_flat)
        v = v / (v.norm(dim=1, keepdim=True) + eps)
        lam = None
        for _ in range(steps):
            gv = (grads_flat * v).sum(dim=1)  # shape (b,)
            Hv = torch.autograd.grad(
                outputs=gv, inputs=state,
                grad_outputs=torch.ones_like(gv),
                retain_graph=True, create_graph=True, allow_unused=True
            )[0]
            if Hv is None:
                Hv = torch.zeros_like(state)
            Hv_flat = Hv.view(b, -1)

            normHv = Hv_flat.norm(dim=1, keepdim=True) + eps
            v = Hv_flat / normHv
            lam = normHv.squeeze(1)  # shape (b,)

        bound_max = 1.999/(safety * lam)
        bound_min = torch.full_like(bound_max, eps)
        return bound_min, bound_max

    def init_state(self, x, y, x_init=None):
        if x_init is not None:
            return x_init
        
        # if self.init_type.upper() == "MNE":
        #     state_0 = self.mnep_init(y)
        if self.init_type.upper() == "NOISE":
            state_0 = self.noise_init(x)
        elif self.init_type.upper() == "ZEROS":
            state_0 = self.zeros_init(x)
        elif self.init_type.upper() == "DIRECT":
            state_0 = self.direct_init(y)
        else:
            raise Exception(f"{self.init_type=} unknown state init type")
        return state_0.detach().requires_grad_(True)

    def noise_init(self, x):
        noise = torch.randn(*x.shape, device=x.device)
        return self.noise_ampl*(noise/noise.max())

    def zero_init(self, x):
        return torch.zeros_like(x, device=x.device)

    def load_init_model(self, init_model):
        if self.init_type.upper() == "DIRECT":      
            self.init_model = init_model
            self.init_model.eval()
            self.init_model.to(device = next(self.parameters()).device)
            for p in self.init_model.parameters():
                p.requires_grad = False
        else:
            print(f"Current init state mode: {self.init_type.upper()}. Please change init state mode: DIRECT")

    def direct_init(self, y):
        with torch.no_grad(): 
            direct_sol = self.init_model( y )
        return direct_sol

    # def mnep_invop(self, mne_info, fwd, method="MNE"): 
    #     """
    #     Compute the inverse operator for the minimum norm solution *method* (e.g MNE) based on the mne-python algorithms.
    #     intput : 
    #     - mne_info : mne-python *info* object associated with the eeg data
    #     - fwd : mne-python forward operator linked with the simulated data (head model)
    #     - method : method to use (MNE, sLORETA, dSPM, eLORETA c.f mne-python documentation on minimum-norm inverse solutions)
    #     output : 
    #     - K : inverse operator (torch.tensor)
    #     """
    #     import mne
    #     from mne.minimum_norm.inverse import (_assemble_kernel,
    #                                           _check_or_prepare)
    #     ## compute a "fake" noise covariance
    #     random_state = np.random.get_state() # avoid changing all random number generation when using MNE init
    #     noise_eeg = mne.io.RawArray(
    #             np.random.randn(len(mne_info['chs']), 600), mne_info, verbose=False
    #         )
    #     np.random.set_state(random_state)
    #     noise_cov = mne.compute_raw_covariance(noise_eeg, verbose=False)
    #     ## compute the inverse operator (K)
    #     inv_op = mne.minimum_norm.make_inverse_operator(
    #         info=mne_info,
    #         forward=fwd,
    #         noise_cov=noise_cov,
    #         loose=0,
    #         depth=0,
    #         verbose=False
    #     )

    #     inv = _check_or_prepare(
    #         inv_op, 1, self.reg, method ,None,False
    #     )
        
    #     K, _, _, _ = _assemble_kernel(
    #             inv, label=None, method=method, pick_ori=None, use_cps=True, verbose=False
    #         )
    #     return torch.from_numpy(K)
    
    # def mnep_init(self, y): 
    #     """  
    #     Inverse problem resolution : estimates x from y, using the inverse operator K (based on mne-python algorithms). 
    #     input : 
    #     - y : eeg data (batch, channel, time)
    #     - method : method to use (MNE, sLORETA, dSPM, eLORETA c.f mne-python documentation on minimum-norm inverse solutions)
    #     """
    #     y = y.float()
    #     if self.inv_op is None: 
    #         self.inv_op = self.mnep_invop( self.mne_info, self.fwd.copy(), self.init_type ).float().to(device = next(self.parameters()).device)
        
    #     return torch.matmul(self.inv_op.to(device = next(self.parameters()).device), y)


class EsiGradSolver_multiA(EsiGradSolver):
    def forward(self, x: torch.Tensor, y: torch.Tensor, steps: int | None = None, return_all: bool = False):
        """
        Run the inner optimization loop and return the final state.

        Args:
            x: input tensor for initialization (or ignored depending on init rule)
            y: target tensor (needed for computing inner loss)
            steps: number of unrolled inner steps
            return_all: whether to return list of inner losses

        Returns:
            final_state or (final_state, inner_losses)
        """
        if steps is None:
            steps = self.n_steps

        with torch.set_grad_enabled(True):
            # Initialize state
            state = self.init_state(x, y)
            self.grad_mod.reset_state(x)

            inner_losses: list[torch.Tensor] = []
            full_terms: dict = {}

            # Unrolled inner loop
            for step in range(steps):
                # compute inner loss
                inner_loss, terms, intermediate = self.inner_loss(x=state, y=y, L=self.fwd.A, AE=self.reg_net, return_terms=True, return_intermediate=True)
                
                inner_losses.append(inner_loss.detach())
                for k, v in terms.items():
                    if k not in full_terms.keys():
                        full_terms[k] = [v.detach(),]
                    else:
                        full_terms[k].append(v.detach())
                # gradient wrt state
                grad = torch.autograd.grad(
                    inner_loss,
                    state,
                    create_graph=True,
                )[0]

                # track grad norm
                self.last_grad_norm = grad.detach().norm()

                # compute bounds (optional)
                if self.bound_estimating_steps is not None:
                    if self.bound_estimating_steps == 0:
                        bounds = self.compute_bounds(intermediate, state, grad)
                    else:
                        bounds = self.estimate_bounds(inner_loss, state, grad, self.bound_estimating_steps)
                else:
                    bounds = None

                # apply grad modifier
                grad = self.grad_mod(state, grad, bounds)

                # apply update rule
                state = self.update_rule(state, grad, step)

                # in eval mode, detach to avoid graph growth
                if not self.training:
                    state = state.detach().requires_grad_(True)

            return (state, inner_losses, full_terms) if return_all else state
