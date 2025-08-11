
import random
import argparse
import contextlib
import torch
import torch.optim as optim
from merger.data_flower import all_h5, H5Dataset
from merger.merger_net import Net
from merger.composed_chamfer import composed_sqrt_chamfer
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from glob import glob

net = Net(2048, 10).cuda()
net.load_state_dict(torch.load("chairmerger.pt", map_location=torch.device("cuda"))['model_state_dict'])
net.eval()

DATASET = "shapenetcorev2_hdf5_2048/train/"
TESTSET = "shapenetcorev2_hdf5_2048/val/"

h5_files = glob(f'{DATASET}**/*.h5', recursive=True)
dataset = H5Dataset(h5_files, normalize=True, include_label=False,subclasses=(14,), )
loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)


h5_files_test = glob(f'{TESTSET}**/*.h5', recursive=True)
dataset_test = H5Dataset(h5_files_test, normalize=True, include_label=False,subclasses=(14,), )
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=8, shuffle=True, num_workers=0)

for batch_x in loader:
    recon, key_points, kpa, emb, null_activation = net(batch_x.cuda())
    # merged = torch.cat(recon, dim=1)  # [B, total_points, 3]

    import pdb; pdb.set_trace()
