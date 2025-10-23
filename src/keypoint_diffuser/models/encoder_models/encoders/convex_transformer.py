import plotly.graph_objects as go
import torch
from plotly.colors import qualitative
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


class ConvexTransformer(nn.Module):
    def __init__(
        self,
        zdim,
        input_dim=3,
        extra_latent=10,
        embed_dim=128,
        num_heads=4,
        num_pos_freqs=32,
        pos_scale=10.0,
    ):
        super().__init__()
        self.num_keypoints = zdim
        self.extra_latent = extra_latent

        # backbone encoder
        self.encoder = PointTransformerV3(in_channels=3, enable_flash=False)

        # positional encoding
        self.pos_enc = FourierPositionalEncoding(
            num_freqs=num_pos_freqs, scale=pos_scale
        )

        # project (feat || posenc) -> attn dim
        in_feat_dim = 64 + self.pos_enc.out_dim
        self.feat_proj = nn.Linear(in_feat_dim, embed_dim)

        # learnable queries for K keypoints
        self.queries = nn.Parameter(torch.randn(zdim, embed_dim))

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        if extra_latent > 0:
            self.extra_latent_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, extra_latent),
            )

    def forward(self, data_dict):
        """
        Returns keypoints as attention-weighted convex combinations of input points.
        """
        point = self.encoder(
            data_dict
        )  # point.feat: (N,64), point.coord: (N,3), point.offset: (B,)

        B = point.offset.shape[0]
        lengths = point.offset.diff(
            prepend=torch.tensor([0], device=point.offset.device)
        )

        # per-point features + Fourier PE
        posenc = self.pos_enc(point.coord)  # (N, 2F)
        feat_cat = torch.cat([point.feat, posenc], dim=-1)  # (N, 64+2F)
        feat_cat = self.feat_proj(feat_cat)  # (N, D)

        # split/pad to batches
        feat_split = feat_cat.split(lengths.tolist(), dim=0)
        coord_split = point.coord.split(lengths.tolist(), dim=0)
        feat_padded = pad_sequence(feat_split, batch_first=True)  # (B, Nmax, D)
        coord_padded = pad_sequence(coord_split, batch_first=True)  # (B, Nmax, 3)

        Nmax = feat_padded.size(1)
        mask = torch.arange(Nmax, device=feat_padded.device).expand(
            B, Nmax
        ) >= lengths.unsqueeze(
            1
        )  # True=pad

        queries = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, K, D)

        # ask MHA for per-head weights
        attn_out, attn_w = self.attn(
            queries,
            feat_padded,
            feat_padded,
            key_padding_mask=mask,
            need_weights=True,
            average_attn_weights=False,
        )  # attn_w: (B, H, K, Nmax) already softmaxed over N

        attn_w = attn_w.mean(dim=1)

        # convex combination: (B,K,N) @ (B,N,3) -> (B,K,3)
        keypoints = torch.bmm(attn_w, coord_padded)

        if self.extra_latent > 0:
            extra_latent = self.extra_latent_proj(
                attn_out.mean(dim=1)
            )  # (B, extra_latent)
            return keypoints, extra_latent

        return keypoints


def make_interactive_attention_html(
    attn_w: torch.Tensor,
    coords: torch.Tensor,
    keypoints: torch.Tensor,
    top_n: int = 100,
    out_path: str = "attn_keypoints.html",
    background_alpha: float = 0.24,
):
    """
    Create a single interactive 3D figure showing all keypoints with distinct colors.
    Returns the output path.
    """
    # detach / to numpy
    A = attn_w.detach().cpu().numpy()  # (K, N)
    P = coords.detach().cpu().numpy()  # (N, 3)
    KPT = keypoints.detach().cpu().numpy()  # (K, 3)

    K, N = A.shape
    top_n = int(min(top_n, N))

    # color palette (repeat if needed)
    palette = qualitative.Dark24
    if len(palette) < K:
        # repeat palette to cover K
        times = (K + len(palette) - 1) // len(palette)
        palette = (palette * times)[:K]
    else:
        palette = palette[:K]

    fig = go.Figure()

    # Background cloud (faint gray)
    fig.add_trace(
        go.Scatter3d(
            x=P[:, 0],
            y=P[:, 1],
            z=P[:, 2],
            mode="markers",
            marker={"size": 2, "opacity": background_alpha, "color": "gray"},
            name="all points",
            showlegend=True,
            legendgroup="background",
        )
    )

    # Per-keypoint top-N + keypoint marker
    for k in range(K):
        w = A[k]  # (N,)
        # top-N indices
        top_idx = np.argpartition(w, -top_n)[-top_n:]
        # sort them for nicer hover (descending weight)
        top_idx = top_idx[np.argsort(-w[top_idx])]
        pts = P[top_idx]
        color = palette[k]

        # top-N points for this k
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker={"size": 4, "opacity": 0.95, "color": color},
                name=f"k{k} top-{top_n}",
                legendgroup=f"k{k}",
                hovertemplate=(
                    f"<b>k{k} top</b><br>"
                    + "x:%{x:.3f}<br>y:%{y:.3f}<br>z:%{z:.3f}<br>"
                    + "w:%{customdata:.5f}<extra></extra>"
                ),
                customdata=w[top_idx],
            )
        )

        # keypoint marker for this k
        fig.add_trace(
            go.Scatter3d(
                x=[KPT[k, 0]],
                y=[KPT[k, 1]],
                z=[KPT[k, 2]],
                mode="markers",
                marker={
                    "size": 10,
                    "symbol": "diamond",
                    "opacity": 1.0,
                    "color": color,
                    "line": {"width": 1, "color": "black"},
                },
                name=f"k{k} (keypoint)",
                legendgroup=f"k{k}",
            )
        )

    fig.update_layout(
        title=f"Interactive attention → keypoints (Top-{top_n} per keypoint)",
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
            "aspectmode": "data",
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )

    # Save a self-contained HTML
    fig.write_html(out_path, include_plotlyjs=True, full_html=True)
    return out_path
