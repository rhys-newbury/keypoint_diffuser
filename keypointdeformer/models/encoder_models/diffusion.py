import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module

from .common import ConcatSquashLinear, FiLMResidualMLP


def init_linear(layer, stddev):
    nn.init.normal_(layer.weight, std=stddev)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0.0)


class VarianceSchedule(Module):
    def __init__(self, num_steps, beta_1, beta_T, mode="linear"):
        super().__init__()
        assert mode in ("linear",)
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)  # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[
                i
            ]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sigmas_flex", sigmas_flex)
        self.register_buffer("sigmas_inflex", sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps + 1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert flexibility >= 0 and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (
            1 - flexibility
        )
        return sigmas


class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, init_scale=0.25):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, k, v):
        # Attention block with residual + norm
        attn_out, _ = self.attn(x, k, v)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Feedforward block with residual + norm
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        point_dim,
        context_dim,
        embed_dim=128,
        num_heads=4,
        n_layers=2,
        dropout=0.1,
        init_scale=1.0,
    ):
        super().__init__()
        self.point_proj = nn.Linear(point_dim, embed_dim)
        self.context_proj = nn.Linear(context_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, point_dim)

        self.layers = nn.ModuleList(
            [
                CrossAttentionLayer(embed_dim, num_heads, dropout, init_scale)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, context):
        """
        x:       (B, N, point_dim)
        context: (B, 1, context_dim)
        """
        # Project inputs
        q = self.point_proj(x)  # (B, N, embed_dim)
        k = self.context_proj(context)  # (B, 1, embed_dim)
        v = k  # context-only keys/values

        for layer in self.layers:
            q = layer(q, k, v)  # (B, N, embed_dim)

        return self.out_proj(q)  # Project back to (B, N, point_dim)


class MLP(nn.Module):
    def __init__(
        self, *, device: torch.device, dtype: torch.dtype, width: int, init_scale: float
    ):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width * 4, width, device=device, dtype=dtype)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class PointwiseNetOld(Module):
    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = nn.ModuleList(
            [
                ConcatSquashLinear(3, 128, context_dim + 3),
                ConcatSquashLinear(128, 256, context_dim + 3),
                ConcatSquashLinear(256, 512, context_dim + 3),
                ConcatSquashLinear(512, 256, context_dim + 3),
                ConcatSquashLinear(256, 128, context_dim + 3),
                ConcatSquashLinear(128, 3, context_dim + 3),
            ]
        )

    def extract_features(self, x, beta, context):
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)  # (B, 1, 1)
        context = context.view(batch_size, 1, -1)  # (B, 1, F)

        time_emb = torch.cat(
            [beta, torch.sin(beta), torch.cos(beta)], dim=-1
        )  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)  # (B, 1, F+3)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) // 2:
                out = self.act(out)

        return out

    def forward(self, x, beta, context):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)  # (B, 1, 1)
        context = context.view(batch_size, 1, -1)  # (B, 1, F)

        time_emb = torch.cat(
            [beta, torch.sin(beta), torch.cos(beta)], dim=-1
        )  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)  # (B, 1, F+3)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out


class PointwiseNet(Module):
    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        init_scale = 0.25

        self.width = 512
        self.time_embed = MLP(
            device="cuda",
            dtype=torch.float32,
            width=self.width,
            init_scale=init_scale * math.sqrt(1.0 / self.width),
        )

        self.ctx_embed = MLP(
            device="cuda",
            dtype=torch.float32,
            width=context_dim,
            init_scale=init_scale * math.sqrt(1.0 / context_dim),
        )

        self.cross_attn = CrossAttentionBlock(
            point_dim=3,
            context_dim=context_dim + self.width,
            embed_dim=128,
            num_heads=8,
            n_layers=12,
            init_scale=init_scale,
        )

        self.layers = nn.ModuleList(
            [
                FiLMResidualMLP(3, 512, context_dim + self.width),
                FiLMResidualMLP(512, 512, context_dim + self.width),
                FiLMResidualMLP(512, 512, context_dim + self.width),
                FiLMResidualMLP(512, 3, context_dim + self.width),
            ]
        )

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].to(timesteps.dtype) * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, x, beta, context):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)  # (B, 1, 1)
        context = context.view(batch_size, 1, -1)  # (B, 1, F)

        # time_emb = torch.cat(
        #     [beta, torch.sin(beta), torch.cos(beta)], dim=-1
        # )  # (B, 1, 3)
        t_embed = self.time_embed(self.timestep_embedding(beta, self.width)).squeeze(2)
        c_project = self.ctx_embed(context)

        ctx_emb = torch.cat([t_embed, c_project], dim=-1)  # (B, 1, F+3)

        out = x + self.cross_attn(x, ctx_emb)

        # out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb.squeeze(1), x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out


class DiffusionPoint(Module):
    def __init__(self, net, var_sched: VarianceSchedule):
        super().__init__()
        self.net = net
        self.freeze_network()
        self.var_sched = var_sched

    def freeze_network(self):
        pass
        # self.frozen_net = copy.deepcopy(self.net)

    def get_loss(self, x_0, context, t=None):
        """
        Args:
            x_0:  Input point cloud, (B, N, d).
            context:  Shape latent, (B, F).
        """
        x_0 = x_0.permute(0, 2, 1)
        batch_size, _, point_dim = x_0.size()
        if t is None:
            t = self.var_sched.uniform_sample_t(batch_size)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)  # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)  # (B, 1, 1)

        e_rand = torch.randn_like(x_0)  # (B, N, d)
        e_theta = self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context)
        loss = F.mse_loss(
            e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction="mean"
        )
        return loss

    def forward(self, x, sigma, context):
        return self.net(x, sigma, context=context)

    def forward_diffuse(self, x_0, epsilon, t):
        """
        Computes forward diffusion process:
        x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * epsilon.
        """
        alpha_bar = self.var_sched.alpha_bars[t].view(-1, 1, 1)
        return torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * epsilon

    def sample(self, num_points, context, point_dim=3, flexibility=0.0, ret_traj=False):
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
        traj = {self.var_sched.num_steps: x_T}
        for t in range(self.var_sched.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.var_sched.betas[[t] * batch_size]
            e_theta = self.net(x_t, beta=beta, context=context)
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t - 1] = x_next.detach()  # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()  # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]

        if ret_traj:
            return traj
        else:
            return traj[0]
