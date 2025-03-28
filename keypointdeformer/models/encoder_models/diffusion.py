import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, ModuleList

from .common import ConcatSquashLinear
from .encoders.PointCloudTransformer.model import PCT


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


class CrossAttentionPointCloud(nn.Module):
    def __init__(self, point_channels, context_dim, output_dim):
        """
        Args:
            point_channels: Number of input channels for the point cloud.
            context_dim: Number of features in the global context vector.
            output_dim: Number of output channels for the final representation.
        """
        super().__init__()

        # Point cloud projections
        self.q_conv = nn.Conv1d(
            point_channels, output_dim, 1, bias=False
        )  # Query from point cloud
        self.k_linear = nn.Linear(
            context_dim, output_dim, bias=False
        )  # Key from context
        self.v_linear = nn.Linear(context_dim, output_dim)  # Value from context
        self.output_dim = output_dim
        # Point cloud's value projection
        self.point_v_conv = nn.Conv1d(
            point_channels, output_dim, 1
        )  # Value from point cloud

        # Output projection
        self.trans_conv = nn.Conv1d(output_dim, output_dim, 1)
        self.after_norm = nn.BatchNorm1d(output_dim)

        self.residual_proj = nn.Conv1d(output_dim, 3, 1)  # Project from 128 to 3

        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, context):
        """
        Args:
            x: Point cloud features, [B, channels, N].
            context: Global context vector, [B, context_dim].

        Returns:
            Updated point cloud features, [B, output_dim, N].
        """
        batch_size, num_points, _ = x.size()

        # Project point cloud (query)
        x_q = self.q_conv(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, N, output_dim]

        # Project context (key, value)
        context_k = self.k_linear(context).unsqueeze(-1)  # [B, output_dim, 1]
        context_v = self.v_linear(context).unsqueeze(-1)  # [B, output_dim, 1]

        # Broadcasting: expand context to match point cloud points
        energy = torch.bmm(x_q, context_k) / (self.output_dim**0.5)  # [B, N, 1]
        attention = self.softmax(energy)  # [B, N, 1]

        # Weighted sum: directly use attention to scale context
        x_v = self.point_v_conv(x.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # [B, N, output_dim]
        aggregated_context = attention * context_v.permute(
            0, 2, 1
        )  # [B, N, output_dim]
        x_s = x_v + aggregated_context  # Combine [B, output_dim, N]

        # Transform output and apply residual connection
        x_s = self.act(
            self.after_norm(self.trans_conv(x_s.permute(0, 2, 1)))
        )  # [B, output_dim, N]
        x_s = self.residual_proj(x_s)  # [B, 3, N]
        x = x + x_s.permute(0, 2, 1)  # Add residual connection

        return x


class PointwiseNetWithAttention(Module):
    def __init__(self, point_dim, context_dim, residual, num_heads):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.attention = CrossAttentionPointCloud(
            point_channels=3, context_dim=context_dim + 3, output_dim=128
        )
        self.layers = ModuleList(
            [
                ConcatSquashLinear(1024, 128, context_dim + 3),
                ConcatSquashLinear(128, 3, context_dim + 3),
            ]
        )
        self.encoder = PCT(samples=[2048, 2048])

    def forward(self, x, beta, context):
        """
        Args:
            x: Point clouds at some timestep t, (B, N, d).
            beta: Time. (B, ).
            context: Shape latents. (B, F).
        """

        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)  # (B, 1, 1)
        context = context.view(batch_size, -1)  # (B, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1).squeeze(
            1
        )  # (B, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)  # (B, d + 3)

        context = ctx_emb.view(batch_size, 1, -1)

        # Cross-Attention between context and x
        att_emb = self.attention(x, ctx_emb)  # Output: (B, d)

        emb, _, _ = self.encoder(att_emb.permute(0, 2, 1))
        emb = emb.permute(0, 2, 1)

        # Concatenate time embedding with context

        # Feed through network layers
        out = emb
        for i, layer in enumerate(self.layers):
            out = layer(ctx=context, x=out)
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
        self.layers = ModuleList(
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


class DiffusionPoint(Module):
    def __init__(self, net, var_sched: VarianceSchedule):
        super().__init__()
        self.net = net
        self.freeze_network()
        self.var_sched = var_sched

    def freeze_network(self):
        pass
        # self.frozen_net = copy.deepcopy(self.net)

    def get_loss(self, x_0, context, use_perceptual_loss=False, t=None):
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
        return loss + (
            0.001 * self.get_perceptual_loss(x_0, context, t=t)
            if use_perceptual_loss
            else 0
        )

    def forward(self, x, sigma, context):
        return self.net(x, sigma, context=context)

    def forward_diffuse(self, x_0, epsilon, t):
        """
        Computes forward diffusion process:
        x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * epsilon.
        """
        alpha_bar = self.var_sched.alpha_bars[t].view(-1, 1, 1)
        return torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * epsilon

    def get_perceptual_loss(self, x_0, context, t=None):
        # x_0 = x_0.permute(0, 2, 1)
        batch_size, _, point_dim = x_0.size()
        if t is None:
            t = self.var_sched.uniform_sample_t(batch_size)

        beta = self.var_sched.betas[t]

        e_rand = torch.randn_like(x_0)
        x_t = self.forward_diffuse(x_0, e_rand, t)

        v_hat_t = self.net(x_t, context=context, beta=beta)

        alpha_bar = self.var_sched.alpha_bars[t]
        sqrt_alpha_bar = torch.sqrt(alpha_bar).view(-1, 1, 1)
        one_minus_sqrt_alpha_bar = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)

        x_0_hat = sqrt_alpha_bar * x_t - one_minus_sqrt_alpha_bar * v_hat_t
        epsilon_hat = sqrt_alpha_bar * v_hat_t + one_minus_sqrt_alpha_bar * x_t

        t_prime = self.var_sched.uniform_sample_t(batch_size)
        x_t_prime = self.forward_diffuse(x_0, e_rand, t_prime)
        x_hat_t_prime = self.forward_diffuse(x_0_hat, epsilon_hat, t_prime)

        beta_prime = self.var_sched.betas[t_prime]

        f1 = self.frozen_net.extract_features(
            x_hat_t_prime, context=context, beta=beta_prime
        )
        f2 = self.frozen_net.extract_features(
            x_t_prime, context=context, beta=beta_prime
        )
        perceptual_loss = F.mse_loss(f1, f2)

        return perceptual_loss

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
