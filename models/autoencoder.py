from collections import OrderedDict
import torch
from torch import nn
from typing import Union, List, Tuple


KernelSize = Union[int, List[int], Tuple[int, ...]]


class BaseAE(nn.Module):
    """
    Generic class for Autoencoder.
    
    Attributes:
        pretrained_model_path: Path to the pretrained weights, if any.
        fixed: Boolean indicating whether the model's weights are fixed or trainable.
    """
    def __init__(self, pretrained_model_path: str = None, fixed: bool = True):
        super().__init__()
        self.build_model()

        # Load pretrained weights if path is available
        self.pretrained_model_path = pretrained_model_path
        if self.pretrained_model_path:
            self.load_state_dict(torch.load(pretrained_model_path))
            print(f"[AE] Loaded pretrained weights from: {pretrained_model_path}")
        
        # Freeze the model if fixed is True
        self.fixed = fixed
        if self.fixed: 
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder."""
        # return self.decoder(self.encoder(x))
        return self.decoder(self.encoder(x))

    def forward_enc(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input x"""
        return self.encoder(x)


class Conv1DAE(BaseAE):
    """
    Conv1D Autoencoder used as a regularizer inside L2O.

    Attributes:
        dim_in: Input dimension (number of input features).
        dim_hidden: Dimension of the hidden layer(s).
        dim_out: Output dimension (number of output features, typically equal to dim_in for reconstruction).
        bias: Whether to use bias terms in the convolutional layers.
        kernel_size: Size of the convolutional kernels (can be a single integer or a list of integers for different layers).
    """
    def __init__(self, dim_in: int = 1284, dim_hidden: int = 128, dim_out: int = 1284, kernel_size: KernelSize = 3, bias: bool = False, pretrained_model_path: str = None, fixed: bool = False):
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.kernel_size = kernel_size
        self.bias = bias
        super().__init__(pretrained_model_path, fixed)

    def build_model(self):
        # Normalize kernel_size
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 3
        elif isinstance(self.kernel_size, (list, tuple)):
            assert len(self.kernel_size) == 3, "kernel_size list must have 3 values"
        else:
            raise TypeError(f"Invalid kernel_size type: {type(self.kernel_size)}")

        self.encoder = nn.Sequential(OrderedDict([
            ('conv_in', nn.Conv1d(in_channels=self.dim_in, out_channels=self.dim_hidden, kernel_size=self.kernel_size[0], bias=self.bias, padding="same")),
            ('relu_in', nn.ReLU()),
            ('conv_hidden', nn.Conv1d(in_channels=self.dim_hidden, out_channels=self.dim_hidden, kernel_size=self.kernel_size[1], bias=self.bias, padding="same")),
            ('relu_hidden', nn.ReLU()),
        ]))

        self.decoder = nn.Sequential(OrderedDict([
            ("conv_out", nn.Conv1d(in_channels=self.dim_hidden, out_channels=self.dim_out, kernel_size=self.kernel_size[2], bias=self.bias, padding="same"))
        ]))


class LConv1DAE(BaseAE):
    """
    Conv1D Autoencoder used as a regularizer inside L2O.

    Attributes:
        dim_in: Input dimension (number of input features).
        dim_hidden: Dimension of the hidden layer(s).
        dim_out: Output dimension (number of output features, typically equal to dim_in for reconstruction).
        bias: Whether to use bias terms in the convolutional layers.
        kernel_size: Size of the convolutional kernels (can be a single integer or a list of integers for different layers).
    """
    def __init__(self, dim_in: int = 1284, dim_hidden: int = 128, dim_out: int = 1284, kernel_size: KernelSize = 3, bias: bool = False, pretrained_model_path: str = None, fixed: bool = False):
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.kernel_size = kernel_size
        self.bias = bias
        super().__init__(pretrained_model_path, fixed)

    def build_model(self):
        # Normalize kernel_size
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 3
        elif isinstance(self.kernel_size, (list, tuple)):
            assert len(self.kernel_size) == 3, "kernel_size list must have 3 values"
        else:
            raise TypeError(f"Invalid kernel_size type: {type(self.kernel_size)}")

        self.encoder = nn.Sequential(OrderedDict([
            ('conv_in', nn.Conv1d(in_channels=self.dim_in, out_channels=self.dim_hidden, kernel_size=self.kernel_size[0], bias=self.bias, padding="same")),
            ('relu_in', nn.ReLU()),
            ('conv_hidden', nn.Conv1d(in_channels=self.dim_hidden, out_channels=self.dim_hidden, kernel_size=self.kernel_size[1], bias=self.bias, padding="same")),
            ('relu_hidden', nn.ReLU()),
        ]))

        self.decoder = nn.Sequential(OrderedDict([
            ("conv_out", nn.Conv1d(in_channels=self.dim_hidden, out_channels=self.dim_out, kernel_size=self.kernel_size[2], bias=self.bias, padding="same"))
        ]))

    def conv2linear(self):
        L = 256 
        self.A_in, b_in = self._conv1d_same_to_sparse_A(self.encoder.conv_in, L)
        self.A_hidden, b_hidden = self._conv1d_same_to_sparse_A(self.encoder.conv_hidden, L)
        self.A_out, b_out = self._conv1d_same_to_sparse_A(self.decoder.conv_out, L)

    def _same_padding_left_right(kernel_size: int, dilation: int = 1):
        # PyTorch "same": total padding = dilation*(k-1)
        total = dilation * (kernel_size - 1)
        left = total // 2
        right = total - left
        return left, right

    def _conv1d_same_to_sparse_A(conv: nn.Conv1d, L: int):
        """
        Build sparse COO A so that vec(y) = A @ vec(x) + b
        for PyTorch Conv1d with padding='same', stride=1, dilation=1.

        Flatten/vec convention:
        vec(x): [c=0 positions 0..L-1, c=1 positions 0..L-1, ...]
        vec(y): same layout for out channels.

        Returns:
        A: sparse COO of shape (C_out*L, C_in*L)
        b: dense vector of shape (C_out*L,)  (bias repeated over positions)
        """
        assert isinstance(conv, nn.Conv1d)
        assert conv.stride == (1,)
        assert conv.dilation == (1,)
        assert conv.padding == "same"

        w = conv.weight  # (C_out, C_in, K)
        C_out, C_in, K = w.shape
        device, dtype = w.device, w.dtype

        pad_left, pad_right = _same_padding_left_right(K, dilation=1)

        rows = []
        cols = []
        vals = []
        i = torch.arange(L, device=device)  # (L,)
        for co in range(C_out):
            row_base = co * L
            for ci in range(C_in):
                col_base = ci * L
                for k in range(K):
                    j = i + (k - pad_left)  # (L,)
                    mask = (j >= 0) & (j < L)
                    if mask.any():
                        rows_k = row_base + i[mask]
                        cols_k = col_base + j[mask]
                        vals_k = torch.full((rows_k.numel(),), w[co, ci, k], device=device, dtype=dtype)

                        rows.append(rows_k)
                        cols.append(cols_k)
                        vals.append(vals_k)

        rows = torch.cat(rows) if rows else torch.empty((0,), device=device, dtype=torch.long)
        cols = torch.cat(cols) if cols else torch.empty((0,), device=device, dtype=torch.long)
        vals = torch.cat(vals) if vals else torch.empty((0,), device=device, dtype=dtype)

        indices = torch.stack([rows, cols], dim=0)  # (2, nnz)

        A = torch.sparse_coo_tensor(
            indices, vals,
            size=(C_out * L, C_in * L),
            device=device, dtype=dtype
        ).coalesce()

        if conv.bias is not None:
            b = conv.bias.repeat_interleave(L)  # (C_out*L,)
        else:
            b = torch.zeros(C_out * L, device=device, dtype=dtype)

        return A, b


class LstmAE(BaseAE):
    """
    LSTM Autoencoder used as a regularizer inside L2O.

    Attributes:
        dim_in: Input dimension (number of input features).
        dim_hidden: Dimension of the hidden layer(s).
        dim_out: Output dimension (number of output features, typically equal to dim_in for reconstruction).
        bias: Whether to use bias terms in the convolutional layers.
        kernel_size: Size of the convolutional kernels (can be a single integer or a list of integers for different layers).
    """
    def __init__(self, dim_in: int = 1284, dim_hidden: int = 128, dim_out: int = 1284, kernel_size: KernelSize = 1, lstm_num_layers: int = 1, lstm_dropout: float = 0.0, batch_first: bool = True, bidirectional: bool = True, bias: bool = False, pretrained_model_path: str = None, fixed: bool = False):
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.kernel_size = kernel_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_dropout = lstm_dropout
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.bias = bias
        super().__init__(pretrained_model_path, fixed)

    def build_model(self):
        # Normalize kernel_size
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 2
        elif isinstance(kernel_size, (list, tuple)):
            assert len(kernel_size) == 2, "kernel_size list must have 2 values"
        else:
            raise TypeError(f"Invalid kernel_size type: {type(kernel_size)}")

        self.conv_encoder = nn.Sequential(OrderedDict([
            ('conv_in', nn.Conv1d(in_channels=self.dim_in, out_channels=self.dim_hidden, kernel_size=kernel_size[0], bias=self.bias, padding="same")),
            ('relu_in', nn.ReLU()),
        ]))

        D = 2 if self.bidirectional else 1
        self.lstm_hidden = nn.LSTM(input_size = self.dim_hidden,
                           hidden_size   = self.dim_hidden,
                           num_layers    = self.lstm_num_layers,
                           bias          = self.bias,
                           dropout       = self.lstm_dropout,
                           batch_first   = self.batch_first,
                           bidirectional = self.bidirectional)

        self.decoder = nn.Sequential(OrderedDict([
            ('relu_hidden', nn.ReLU()),
            ("conv_out", nn.Conv1d(in_channels=D*self.dim_hidden, out_channels=self.dim_out, kernel_size=kernel_size[1], bias=self.bias, padding="same"))
        ]))

    def encoder(self, x):
        x = self.conv_encoder(x)
        x = x.transpose(-1, -2)
        x, (h, c) = self.lstm_hidden(x)
        return x.transpose(-1, -2)