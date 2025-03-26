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
        # batch_size, _, num_points = point_cloud_batch.shape

        # # Rearrange to [B, N, 3] and then flatten to [B*N, 3]
        # coords = point_cloud_batch.permute(0, 2, 1).reshape(-1, 3)

        # # Use the coordinates as features (copy to be explicit)
        # feats = coords.clone()

        # # Create the offset tensor. For each instance, we know the start index:
        # # 0, num_points, 2*num_points, ..., B*num_points
        # offset = torch.arange(0, (batch_size + 1) * num_points, step=num_points, dtype=torch.int32)

        # data_dict = {
        #     "coord": coords,
        #     "feat": feats,
        #     "offset": offset,
        # }
        point = self.encoder(data_dict)
        point.feat = torch_scatter.segment_csr(
            src=point.feat,
            indptr=nn.functional.pad(point.offset, (1, 0)),
            reduce="mean",
        )
        # import pdb; pdb.set_trace()

        return self.linear(point.feat)


class PointTransformer(nn.Module):
    def __init__(self, zdim, input_dim=3, extra_latent=10):
        super().__init__()

        self.encoder = PCT()

        # self.zdim = zdim
        # self.conv1 = nn.Conv1d(input_dim, 128, 1)
        # self.conv2 = nn.Conv1d(128, 128, 1)
        # self.conv3 = nn.Conv1d(128, 256, 1)
        # self.conv4 = nn.Conv1d(256, 512, 1)
        # self.bn1 = nn.BatchNorm1d(128)
        # self.bn2 = nn.BatchNorm1d(128)
        # self.bn3 = nn.BatchNorm1d(256)
        # self.bn4 = nn.BatchNorm1d(512)

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
        # x = x.transpose(1, 2)
        _, x, _ = self.encoder(x)

        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        # x = self.bn4(self.conv4(x))
        # x = torch.max(x, 2, keepdim=True)[0]
        # x = x.view(-1, 512)

        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)
        # bs = m.shape[0]
        # print(m.shape)
        # import pdb; pdb.set_trace()
        # m = m[:, :-10].reshape(bs, -1, 3)
        v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        v = self.fc3_v(v)
        # v = v.reshape(bs, -1, 3)

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m, v
