import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .PointTransformer.model import PointTransformerV3


class FourierPositionalEncoding(nn.Module):
    def __init__(self, num_freqs=32, scale=10.0):
        super().__init__()
        self.register_buffer("freqs", torch.randn(3, num_freqs) * scale)
        self.out_dim = num_freqs * 2

    def forward(self, xyz):  # xyz: [N, 3]
        proj = 2 * torch.pi * xyz @ self.freqs  # [N, F]
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # [N, 2F]


class PointTransformerv2(nn.Module):
    def __init__(self, zdim, input_dim=3, extra_latent=10, embed_dim=128, num_heads=4):
        super().__init__()
        self.num_keypoints = zdim
        self.extra_latent = extra_latent

        self.encoder = PointTransformerV3(in_channels=3, enable_flash=False)
        self.pos_enc = FourierPositionalEncoding(num_freqs=32, scale=10.0)

        self.feat_proj = nn.Linear(64 + self.pos_enc.out_dim, embed_dim)

        self.queries = nn.Parameter(torch.randn(zdim, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.coord_head = nn.Linear(embed_dim, input_dim)

        if extra_latent > 0:
            self.extra_latent_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, extra_latent),
            )

    def forward(self, data_dict):
        point = self.encoder(
            data_dict
        )  # point.feat: [N, 64], point.pos: [N, 3], point.offset: [B]
        B = point.offset.shape[0]
        point.feat.shape[0]

        # Positional encoding
        pos_enc = self.pos_enc(point.coord)  # [N, D]
        feat = torch.cat([point.feat, pos_enc], dim=-1)
        feat = self.feat_proj(feat)  # [N, embed_dim]

        # Build batch index per point
        torch.repeat_interleave(
            torch.arange(B, device=feat.device),
            point.offset.diff(prepend=torch.tensor([0], device=feat.device)),
        )  # [N]

        # Group into padded batch
        lengths = point.offset.diff(
            prepend=torch.tensor([0], device=feat.device)
        )  # [B]
        feat_split = feat.split(lengths.tolist(), dim=0)  # list of [Li, D]
        feat_padded = pad_sequence(feat_split, batch_first=True)  # [B, Lmax, D]

        # Attention mask: True = ignore
        max_len = feat_padded.size(1)
        mask = torch.arange(max_len, device=feat.device).expand(
            B, max_len
        ) >= lengths.unsqueeze(
            1
        )  # [B, Lmax]

        queries = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]
        attn_out, _ = self.attn(
            queries, feat_padded, feat_padded, key_padding_mask=mask
        )  # [B, K, D]

        keypoints = self.coord_head(attn_out)  # [B, K, 3]

        if self.extra_latent > 0:
            extra_latent = self.extra_latent_proj(
                attn_out.mean(dim=1)
            )  # [B, extra_latent]
            return keypoints, extra_latent

        return keypoints
