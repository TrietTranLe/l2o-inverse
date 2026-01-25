import einops
import torch
from torch import nn


class ConvLstmGradMod(nn.Module):
    """
    Wrapper around the base grad mod that allows for reshaping of the input batch
    Used to convert the lorenz timeseries into an "image" for reuse of conv2d layers
    """
    def __init__(self, dim_hidden, dropout=0.1, downsamp=None, rearrange_from='b c t', rearrange_to='b c t ()', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rearrange_bef = rearrange_from + ' -> ' + rearrange_to
        self.rearrange_aft = rearrange_to + ' -> ' + rearrange_from

        self.dim_hidden = dim_hidden
        self.dropout = torch.nn.Dropout(dropout)
        self._state = []
        self.down = nn.AvgPool2d(downsamp) if downsamp is not None else nn.Identity()
        self.up = (
            nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else nn.Identity()
        )

    def reset_state(self, inp):
        inp = einops.rearrange(inp, self.rearrange_bef)
        size = [inp.shape[0], self.dim_hidden, *inp.shape[-2:]]
        self._grad_norm = None
        self._state = [
            self.down(torch.zeros(size, device=inp.device)),
            self.down(torch.zeros(size, device=inp.device)),
        ]


class x_grad_mod_mul(ConvLstmGradMod):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, dropout=0.1, downsamp=None, rearrange_from='b c t', rearrange_to='b c t ()', *args, **kwargs):
        super().__init__(dim_hidden, dropout, downsamp, rearrange_from, rearrange_to, *args, **kwargs)

        self.encoder = torch.nn.Conv2d(dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2)
        self.decoder = torch.nn.Conv2d(dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2)

        self.encoder_grad = torch.nn.Conv2d(dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2)
        self.decoder_grad = torch.nn.Conv2d(dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2)

        self.gates = torch.nn.Conv2d(
            3 * dim_hidden,
            4 * dim_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.gates_grad = torch.nn.Conv2d(
            2 * dim_hidden,
            4 * dim_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def reset_state(self, inp):
        inp = einops.rearrange(inp, self.rearrange_bef)
        size = [inp.shape[0], self.dim_hidden, *inp.shape[-2:]]
        self._grad_norm = None
        
        self._state = [
            self.down(torch.zeros(size, device=inp.device)),
            self.down(torch.zeros(size, device=inp.device)),
        ]

        self._state_grad = [
            self.down(torch.zeros(size, device=inp.device)),
            self.down(torch.zeros(size, device=inp.device)),
        ]

    def forward(self, x, grad_x, bounds=None):
        x = einops.rearrange(x, self.rearrange_bef)
        grad_x = einops.rearrange(grad_x, self.rearrange_bef)

        if self._grad_norm is None:
            self._grad_norm = (grad_x**2).mean().sqrt()
        grad_x =  grad_x / self._grad_norm

        x_fea = self.dropout(x)
        x_fea = self.down(x_fea)
        x_fea = self.encoder(x_fea)

        grad_x_fea = self.dropout(grad_x)
        grad_x_fea = self.down(grad_x_fea)
        grad_x_fea = self.encoder_grad(grad_x_fea)
        
        grad_x_fea = self._forward_grad(grad_x_fea)
        P = self._forward_P(x_fea, grad_x_fea)
        
        grad_x_fea = self.up(self.decoder_grad(grad_x_fea))
        P = self.up(self.decoder(P))
        P = P*grad_x_fea
        if bounds is not None:
            P_min, P_max = bounds
            P = self._clamp_P(P, P_min, P_max)
        
        # gradients[-1].append((P).detach().to('cpu').squeeze().numpy())
        # gradients[-1].append(grad_x.detach().to('cpu').squeeze().numpy())
        out = P*grad_x
        # gradients[-1].append(out.detach().to('cpu').squeeze().numpy())
        return einops.rearrange(out, self.rearrange_aft), einops.rearrange(P, self.rearrange_aft)

    def _clamp_P(self, P, P_min=None, P_max=None):
        """
        Clamp preconditioner P with flexible bounds:
        - If both P_min and P_max are None -> return P unchanged
        - If only P_min is provided -> clamp lower bound
        - If only P_max is provided -> clamp upper bound
        - If both provided → clamp both sides
        """

        # If nothing to clamp
        if P_min is None and P_max is None:
            return P

        # Expand logic
        def expand_bound_to_P(bound, P):
            """
            Expands bound so it can broadcast with P.
            """
            needed_dims = P.ndim - bound.ndim
            assert needed_dims >= 0, (
                f"Cannot broadcast bound (ndim={bound.ndim}) to P (ndim={P.ndim})"
            )
            shape = (1,) * needed_dims + tuple(bound.shape)
            return bound.view(*shape)

        # Lower bound only
        if P_min is not None and P_max is None:
            P_min = expand_bound_to_P(
                torch.as_tensor(P_min, device=P.device, dtype=P.dtype), P
            )
            return torch.maximum(P, P_min)

        # Upper bound only
        if P_min is None and P_max is not None:
            P_max = expand_bound_to_P(
                torch.as_tensor(P_max, device=P.device, dtype=P.dtype), P
            )
            return torch.minimum(P, P_max)

        # Both bounds
        P_min = expand_bound_to_P(
            torch.as_tensor(P_min, device=P.device, dtype=P.dtype), P
        )
        P_max = expand_bound_to_P(
            torch.as_tensor(P_max, device=P.device, dtype=P.dtype), P
        )
        return torch.clamp(P, P_min, P_max)

    def _forward_P(self, x, grad_x):
        hidden, cell = self._state
        gates = self.gates(torch.cat((x, grad_x, hidden), 1))

        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        in_gate, remember_gate, out_gate = map(
            torch.sigmoid, [in_gate, remember_gate, out_gate]
        )
        cell_gate = torch.tanh(cell_gate)
        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self._state = hidden, cell
        return hidden

    def _forward_grad(self, grad_x):
        hidden, cell = self._state_grad
        gates = self.gates_grad(torch.cat((grad_x, hidden), 1))
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        in_gate, remember_gate, out_gate = map(
            torch.sigmoid, [in_gate, remember_gate, out_gate]
        )
        cell_gate = torch.tanh(cell_gate)
        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self._state_grad = hidden, cell
        return hidden