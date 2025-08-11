

from eval_metrics import EMD_CD_recon 
from glob import glob
import argparse
import torch
import numpy as np
import merger.merger_net as merger_net
from H5Dataset import H5Dataset
from tqdm import tqdm
arg_parser = argparse.ArgumentParser(description="Combined prediction and evaluation for Skeleton Merger.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Prediction args
arg_parser.add_argument('-b', '--batch', type=int, default=8,
                        help='Batch size for prediction.')
arg_parser.add_argument('--max-points', type=int, default=2048,
                        help='Max number of points in each point cloud.')
arg_parser.add_argument('-m', '--checkpoint-path', type=str, default='txt_chair31.pth',
                        help='Model checkpoint file path.')
arg_parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='PyTorch device (e.g., "cuda" or "cpu").')

def normalize_point_clouds(pcs, mode):
    if mode is None:
        print("Will not normalize point clouds.")
        return pcs
    print(f"Normalization mode: {mode}")
    for i in tqdm(range(pcs.size(0)), desc="Normalize"):
        pc = pcs[i]
        if mode == "shape_unit":
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif mode == "shape_bbox":
            pc_max, _ = pc.max(dim=0, keepdim=True)  # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True)  # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        pc = (pc - shift) / scale
        pcs[i] = pc
    return pcs

if __name__ == '__main__':
    ns = arg_parser.parse_args()
    
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    net = merger_net.Net(2048, 10).to("cuda")
    net.load_state_dict(torch.load(ns.checkpoint_path, map_location=torch.device(ns.device))['model_state_dict'])
    net.eval()
    net.cuda()

    TESTSET="/data/shapenetcorev2_hdf5_2048/val"

    h5_files_test = glob(f'{TESTSET}**/*.h5', recursive=True)
    dataset_test = H5Dataset(h5_files_test, normalize=True, include_label=False,subclasses=(14,), )
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=ns.batch, shuffle=True, num_workers=4)


    generated = []
    originals = []

    with torch.no_grad():
        for batch_x in tqdm(loader_test, desc="Evaluating on test set"):
            batch_x = batch_x.cuda()  # (B, N, 3)
            keypoints, recon = net(batch_x, train=False)  # recon: (B, N, 3)

            generated.append(recon.cpu())
            originals.append(batch_x.cpu())

    # Stack into full tensors
    generated = torch.cat(generated, dim=0)     # [N, P, 3]
    originals = torch.cat(originals, dim=0)     # [N, P, 3]

    # Normalize both sets
    generated = normalize_point_clouds(generated, "shape_bbox")
    originals = normalize_point_clouds(originals, "shape_bbox")

    # Compute Chamfer and EMD
    results = EMD_CD_recon(generated.cuda(), originals.cuda(), batch_size=8)
    print(results)
