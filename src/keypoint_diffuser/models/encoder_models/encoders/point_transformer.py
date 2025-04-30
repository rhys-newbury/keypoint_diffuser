import torch_scatter
from torch import nn

from .PointTransformer.model import PointTransformerV3


class PointTransformerv2(nn.Module):
    def __init__(self, zdim, input_dim=3, extra_latent=10):
        super().__init__()

        self.encoder = PointTransformerV3(in_channels=3, enable_flash=False)
        self.linear = nn.Linear(64, zdim * input_dim + extra_latent)

    def forward(self, data_dict):
        point = self.encoder(data_dict)
        point.feat = torch_scatter.segment_csr(
            src=point.feat,
            indptr=nn.functional.pad(point.offset, (1, 0)),
            reduce="mean",
        )

        return self.linear(point.feat)
