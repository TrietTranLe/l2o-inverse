import torch 
from torch import nn
import numpy as np


class EsiGradSolver(nn.Module): 
    def __init__(self, reg_net, grad_mod, update_rule, inner_loss, n_steps, fwd, noise_ampl=1e-3, init_type = "noise", mne_info=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_net = reg_net
        self.grad_mod = grad_mod
        self.update_rule = update_rule
        self.inner_loss = inner_loss
        self.n_steps = n_steps

        self.fwd = fwd
        self.mne_info = mne_info
        self.inv_op = None 
        self.init_type = init_type
        self.reg = 1/25
        self.noise_ampl = noise_ampl

        self._grad_norm = None

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        steps: int | None = None,
        return_all: bool = False,
    ):
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

                # apply grad modifier
                grad = self.grad_mod(state, grad, inner_loss)

                # apply update rule
                state = self.update_rule(state, grad, step)

                # in eval mode, detach to avoid graph growth
                if not self.training:
                    state = state.detach().requires_grad_(True)

            return (state, inner_losses, full_terms) if return_all else state


    def init_state(self, x, y, x_init=None):
        if x_init is not None:
            return x_init
        
        if self.init_type.upper() == "MNE":
            state_0 = state_0 = self.mnep_init(y)
        elif self.init_type.upper() == "NOISE":
            state_0 = self.noise_init(x)
        elif self.init_type.upper() == "ZEROS":
            state_0 = self.zeros_init(x)
        elif self.init_type.upper() == "DIRECT":
            state_0 = self.direct_init(y)
        else:
            raise Exception(f"{self.init_type=} unknown state init type")
        return state_0.detach().requires_grad_(True)

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

    def mnep_invop(self, mne_info, fwd, method="MNE"): 
        """
        Compute the inverse operator for the minimum norm solution *method* (e.g MNE) based on the mne-python algorithms.
        intput : 
        - mne_info : mne-python *info* object associated with the eeg data
        - fwd : mne-python forward operator linked with the simulated data (head model)
        - method : method to use (MNE, sLORETA, dSPM, eLORETA c.f mne-python documentation on minimum-norm inverse solutions)
        output : 
        - K : inverse operator (torch.tensor)
        """
        import mne
        from mne.minimum_norm.inverse import (_assemble_kernel,
                                              _check_or_prepare)
        ## compute a "fake" noise covariance
        random_state = np.random.get_state() # avoid changing all random number generation when using MNE init
        noise_eeg = mne.io.RawArray(
                np.random.randn(len(mne_info['chs']), 600), mne_info, verbose=False
            )
        np.random.set_state(random_state)
        noise_cov = mne.compute_raw_covariance(noise_eeg, verbose=False)
        ## compute the inverse operator (K)
        inv_op = mne.minimum_norm.make_inverse_operator(
            info=mne_info,
            forward=fwd,
            noise_cov=noise_cov,
            loose=0,
            depth=0,
            verbose=False
        )

        inv = _check_or_prepare(
            inv_op, 1, self.reg, method ,None,False
        )
        
        K, _, _, _ = _assemble_kernel(
                inv, label=None, method=method, pick_ori=None, use_cps=True, verbose=False
            )
        return torch.from_numpy(K)
    
    def mnep_init(self, y): 
        """  
        Inverse problem resolution : estimates x from y, using the inverse operator K (based on mne-python algorithms). 
        input : 
        - y : eeg data (batch, channel, time)
        - method : method to use (MNE, sLORETA, dSPM, eLORETA c.f mne-python documentation on minimum-norm inverse solutions)
        """
        y = y.float()
        if self.inv_op is None: 
            self.inv_op = self.mnep_invop( self.mne_info, self.fwd.copy(), self.init_type ).float().to(device = next(self.parameters()).device)
        
        return torch.matmul(self.inv_op.to(device = next(self.parameters()).device), y)

    def noise_init(self, x):
        noise = torch.randn(*x.shape, device=x.device)
        return self.noise_ampl*(noise/noise.max())

    def zero_init(self, x):
        return torch.zeros_like(x, device=x.device)