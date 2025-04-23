import torch.nn.functional as F
import torch_scatter
from torch import nn

from .PointCloudTransformer.model import PCT
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


class PointTransformer(nn.Module):
    def __init__(self, zdim, input_dim=3, extra_latent=10):
        super().__init__()

        self.encoder = PCT()

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(1024, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim * 3 + extra_latent)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)

        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(1024, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, zdim * 3 + extra_latent)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        _, x, _ = self.encoder(x)

        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)

        v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        v = self.fc3_v(v)

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m, v
