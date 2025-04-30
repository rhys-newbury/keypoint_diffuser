import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module
from torch.optim.lr_scheduler import LambdaLR


def weight_init(shape, mode, fan_in, fan_out):
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = {
            "mode": init_mode,
            "fan_in": in_features,
            "fan_out": out_features,
        }
        self.weight = torch.nn.Parameter(
            weight_init([out_features, in_features], **init_kwargs) * init_weight
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias)
            if bias
            else None
        )

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


class FiLMResidualMLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx, hidden_dim=512):
        super().__init__()

        # Feature projection
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim_out)

        # Context modulation
        self.film = nn.Linear(dim_ctx, 2 * hidden_dim)

        # Normalization
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Shortcut if needed
        self.shortcut = (
            nn.Identity() if dim_in == dim_out else nn.Linear(dim_in, dim_out)
        )

    def forward(self, x, ctx):
        """
        x: (B, N, D)
        ctx: (B, C) or (B, 1, C)
        """
        B, N, _ = x.shape

        # Apply FiLM to hidden layer
        gamma, beta = self.film(ctx).chunk(2, dim=-1)  # (B, H), (B, H)
        gamma = gamma.unsqueeze(1)  # (B, 1, H)
        beta = beta.unsqueeze(1)  # (B, 1, H)

        out = self.linear1(x)  # (B, N, H)
        out = self.norm1(out)
        out = F.leaky_relu(gamma * out + beta)

        out = self.linear2(out)  # (B, N, D_out)
        return self.shortcut(x) + out


class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx, **kwargs):
        super().__init__()
        self._layer = Linear(dim_in, dim_out, **kwargs)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False, **kwargs)
        self._hyper_gate = Linear(dim_ctx, dim_out, **kwargs)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        ret = self._layer(x) * gate + bias
        return ret


def get_linear_scheduler(optimizer, start_epoch, end_epoch, start_lr, end_lr):
    def lr_func(epoch):
        if epoch <= start_epoch:
            return 1.0
        elif epoch <= end_epoch:
            total = end_epoch - start_epoch
            delta = epoch - start_epoch
            frac = delta / total
            return (1 - frac) * 1.0 + frac * (end_lr / start_lr)
        else:
            return end_lr / start_lr

    return LambdaLR(optimizer, lr_lambda=lr_func)
