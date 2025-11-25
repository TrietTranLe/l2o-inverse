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