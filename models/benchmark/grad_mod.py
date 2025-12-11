import einops
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import math


class ConvLstmGradMod(nn.Module):
    """
    Wrapper around the base grad model that allows for reshaping of the input batch
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


class ConvLstmGradMod_4DVarNet(ConvLstmGradMod):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, dropout=0.1, downsamp=None, rearrange_from='b c t', rearrange_to='b c t ()', *args, **kwargs):
        super().__init__(dim_hidden, dropout, downsamp, rearrange_from, rearrange_to, *args, **kwargs)

        self.gates = torch.nn.Conv2d(
            dim_in + dim_hidden,
            4 * dim_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.conv_out = torch.nn.Conv2d(
            dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2
        )

    def forward(self, x, grad_x, inner_loss=None):
        grad_x_ori = grad_x.clone()
        grad_x = einops.rearrange(grad_x, self.rearrange_bef)

        if self._grad_norm is None:
            self._grad_norm = (grad_x**2).mean().sqrt()
        grad_x =  grad_x / self._grad_norm
        # grad_x = grad_x/((grad_x**2).mean().sqrt())

        hidden, cell = self._state
        grad_x = self.dropout(grad_x)
        grad_x = self.down(grad_x)
        gates = self.gates(torch.cat((grad_x, hidden), 1))

        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate, remember_gate, out_gate = map(
            torch.sigmoid, [in_gate, remember_gate, out_gate]
        )
        cell_gate = torch.tanh(cell_gate)

        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self._state = hidden, cell
        out = self.conv_out(hidden)
        out = self.up(out)
        
        return einops.rearrange(out, self.rearrange_aft), grad_x_ori
    

class MetaCurvature(nn.Module):
    def __init__(self, dim_out, dim_in, dim_f):
        super().__init__()
        self.dim_out, self.dim_in = dim_out, dim_in
        self.dim_f = dim_f

        # Learnable curvature matrices
        self.M_o = nn.Parameter(torch.eye(self.dim_out))   # (dim_out, dim_out)
        self.M_i  = nn.Parameter(torch.eye(self.dim_in))    # (dim_in, dim_in)
        self.M_f = nn.Parameter(torch.eye(self.dim_f))      # (dim_f, dim_f)

    # def forward(self, G):
    #     """
    #     Apply meta-curvature to gradient tensor of shape:
    #     G: (C_out, C_in, k_h, k_w)
    #     """

    #     # --- output transform ---
    #     G = self.M_out @ G

    #     # --- input transform ---
    #     G = einops.rearrange(G, 'b c t -> b t c')
    #     G = self.M_in @ G

    #     # --- kernel transform ---
    #     G = einops.rearrange(G, 'b t c -> b 1 (t c)')
    #     G = G @ self.M_ker
    #     return einops.rearrange(G, 'b 1 (t c) -> b c t)', t=self.dim_in, c=self.dim_out)
    
    def forward(self, x, G, inner_loss=None):
        """
        Apply meta-curvature to gradient tensor of shape:
        G: (o, i)
        """
        
        G = einops.rearrange(G, 'b o i -> b 1 (o i)')
        G = self.M_f @ G
        
        G = einops.rearrange(G, 'b 1 (o i) -> b i o', i=self.dim_in, o=self.dim_out)
        G = self.M_i @ G
        
        G = einops.rearrange(G, 'b i o -> b o i')
        G = self.M_o @ G
        return G

    def reset_state(self, inp):
        self._grad_norm = None
        

class ModGrad(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_filters=64, kernel_size=1, num_plastic=300, num_mix=5):
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        # self.num_plastic = num_plastic
        self.num_plastic = dim_hidden*dim_hidden
        self.num_mix = num_mix

        self.uu_3 = nn.Sequential(nn.Linear(self.num_plastic, self.num_filters * self.kernel_size),
                                  nn.ReLU(),
                                  nn.Linear(self.num_filters * self.kernel_size, self.num_mix+ self.num_mix*self.num_filters * self.kernel_size)
                                  )
        self.vv_3 = nn.Sequential(nn.Linear(self.num_plastic, self.num_filters * self.kernel_size),
                                  nn.ReLU(),
                                  nn.Linear(self.num_filters * self.kernel_size,
                                            self.num_mix + self.num_mix * self.num_filters * self.kernel_size)
                                  )
        
        if self.kernel_size == 1:
            self.decoder = nn.Conv1d(self.num_filters * self.kernel_size, dim_in, kernel_size=kernel_size, padding="same")
        elif self.kernel_size == 2:
            self.decoder = nn.Sequential(nn.Conv1d(self.num_filters * self.kernel_size, 256, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(),
                                         nn.Conv1d(256, dim_in, kernel_size=1, padding="same"))

        # self.bb_3 = nn.Sequential(nn.Linear(self.num_plastic, self.num_filters),
        #                           nn.ReLU(),
        #                           nn.Linear(self.num_filters,  self.num_mix + self.num_mix* self.num_filters)
        #                           )

        # self.uu_4 = nn.Sequential(nn.Linear(self.num_plastic, self.num_filters * self.kernel_size),
        #                           nn.ReLU(),
        #                           nn.Linear(self.num_filters * self.kernel_size,
        #                                     self.num_mix + self.num_mix *self.num_filters * self.kernel_size)
        #                           )
        # self.vv_4 = nn.Sequential(nn.Linear(self.num_plastic, self.num_filters * self.kernel_size),
        #                           nn.ReLU(),
        #                           nn.Linear(self.num_filters * self.kernel_size,
        #                                     self.num_mix + self.num_mix *self.num_filters * self.kernel_size)
        #                           )
        # self.bb_4 = nn.Sequential(nn.Linear(self.num_plastic, self.num_filters),
        #                           nn.ReLU(),
        #                           nn.Linear(self.num_filters, self.num_mix+ self.num_mix*self.num_filters)
        #                           )
        for param in self.parameters():
            self.init_layer(param)

    def init_layer(self, L):
        # Initialization using fan-in
        if isinstance(L, nn.Conv1d):
            n = L.kernel_size * L.out_channels
            L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
        # elif isinstance(L, nn.BatchNorm2d):
        #     L.weight.data.fill_(1)
        #     L.bias.data.fill_(0)
        # elif isinstance(L, nn.BatchNorm2d):
        #     L.weight.data.fill_(1)
        #     L.bias.data.fill_(0)
        elif isinstance(L, nn.Linear):
            torch.nn.init.kaiming_uniform_( L.weight, nonlinearity='linear')

    def _compute_mod_grad(self):
        if self.num_mix <= 1:
            # conv3_uv, conv3_b = self.assemble_w_b(self.uu_3, self.vv_3, self.bb_3, self._context_params)
            # conv4_uv, conv4_b = self.assemble_w_b(self.uu_4, self.vv_4, self.bb_4, self._context_params)
            conv3_uv = self.assemble_w_b(self.uu_3, self.vv_3, self._context_params)
        else:
            # conv3_uv, conv3_b = self.assemble_w_b_multi(self.uu_3, self.vv_3, self.bb_3, self._context_params)
            # conv4_uv, conv4_b = self.assemble_w_b_multi(self.uu_4, self.vv_4, self.bb_4, self._context_params)
            conv3_uv = self.assemble_w_b_multi(self.uu_3, self.vv_3, self._context_params)

        # self._mod_grad = [conv3_uv, conv3_b, conv4_uv, conv4_b]
        return self.decoder(conv3_uv)

    def reset_state(self, inp):
        self._context_params = torch.zeros(size=[self.num_plastic], device=inp.device)
        self._context_params.requires_grad = True
        self.M_0 = self._compute_mod_grad()
        self._grad_norm = None

    def forward(self, x, G, context_loss):
        # update context params
        self._context_params = -torch.autograd.grad(context_loss, self._context_params, create_graph=True, retain_graph=True)[0]
        
        # compute modulated gradient
        M_k = self._compute_mod_grad()
        
        # reset context params
        self._reset_context_params()
        return M_k*G
    
    def _reset_context_params(self):
        self._context_params = self._context_params.detach()*0.0
        self._context_params.requires_grad = True
        self.M_0 = self._compute_mod_grad()

    def assemble_w_b(self, uu_func, vv_func, lat):
        uu = uu_func(lat)
        vv = vv_func(lat)
        # bb = bb_func(lat)
        wu_ext = uu.unsqueeze(-1)
        wv_ext_t = vv.unsqueeze(-1).transpose(0, 1)
        conv_uv = torch.mm(wu_ext, wv_ext_t)
        # conv_b = bb
        return F.relu(conv_uv) #, F.relu(conv_b)

    def assemble_w_b_multi(self, uu_func, vv_func, lat):

        uu_all = uu_func(lat)
        vv_all = vv_func(lat)
        # bb_all = bb_func(lat)

        mixture_coeff_uu = F.softmax(uu_all[:self.num_mix])
        mixture_coeff_vv = F.softmax(vv_all[:self.num_mix])
        # mixture_coeff_bb = F.softmax(bb_all[:self.num_mix])

        uu = uu_all[self.num_mix:].view(self.num_mix, -1)
        uu = uu * mixture_coeff_uu.unsqueeze(-1)
        uu = uu.sum(0)

        vv = vv_all[self.num_mix:].view(self.num_mix, -1)
        vv = vv * mixture_coeff_vv.unsqueeze(-1)
        vv = vv.sum(0)

        # bb = bb_all[self.num_mix:].view(self.num_mix, -1)
        # bb = bb * mixture_coeff_bb.unsqueeze(-1)
        # bb = bb.sum(0)

        wu_ext = uu.unsqueeze(-1)
        wv_ext_t = vv.unsqueeze(-1).transpose(0, 1)

        conv_uv = torch.mm(wu_ext, wv_ext_t)
        # conv_b = bb

        return F.relu(conv_uv) #, F.relu(conv_b)
    


