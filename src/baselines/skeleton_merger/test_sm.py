import argparse
from pathlib import Path

import numpy as np
import torch
from baselines.skeleton_merger import merger_net
from baselines.test_base import TestBase


class SM(TestBase):
    @staticmethod
    def get_parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument(
            "--max-points",
            type=int,
            default=2048,
            help="Indicates maximum points in each input point cloud.",
        )
        return p

    def load_model(self, model_path: Path, cfg):
        self.model = merger_net.Net(
            cfg.max_points, cfg.key_point
        ).cuda()  # unsupervised network
        self.model.load_state_dict(
            torch.load(model_path)["model_state_dict"]
        )  # load weights
        self.model.eval()  # optional: set to evaluation mode

    def get_keypoints(self, pcd: np.ndarray) -> np.ndarray:
        data = (torch.Tensor(pcd["orig"])).cuda()
        # item 2 is keypoints
        return self.model.get_keypoints(data).cpu().numpy()

    @torch.no_grad()
    def get_reconstruction(
        self,
        pcd: np.ndarray,
        min_points: int = 2048,
    ):
        """
        Greedy reconstruction by activation score:
        - For each item b, sort part indices by MA[b] descending
        - Accumulate corresponding RPCD parts until total points >= min_points
        - If overshoot: downsample to exactly min_points (no replacement)
        - If not enough even after all parts: upsample with replacement to min_points

        Returns:
            reconstructions: list length B; each item is [ merged_(min_points x 3) ]
            data           : torchified input (B, ..., 3)
            kps            : keypoints returned by the model
        """
        device = next(self.model.parameters()).device
        data = torch.as_tensor(pcd, dtype=torch.float32, device=device)

        # RPCD: list[P] of tensors shaped [B, m_i, 3]
        # MA   : [B, P] activation scores per part
        RPCD, kps, _, _, MA = self.model(data)

        B = data.shape[0]
        MA.shape[1]  # number of parts
        reconstructions = []

        for b in range(B):
            # sort parts by activation score (desc)
            scores = MA[b]  # (P,)
            order = torch.argsort(scores, descending=True)
            parts_points = []
            total = 0

            for idx in order.tolist():
                pts = RPCD[idx][b]  # [m_i, 3]
                if pts.numel() == 0:
                    continue
                parts_points.append(pts)
                total += pts.shape[0]
                if total >= min_points:
                    break

            if not parts_points:
                # no parts available for this item; return empty candidate
                reconstructions.append([])
                continue

            merged = torch.cat(parts_points, dim=0)  # [M, 3]
            M = merged.shape[0]

            if min_points <= M:
                # downsample without replacement
                sel = torch.randperm(M, device=merged.device)[:min_points]
                merged = merged[sel]

                reconstructions.append([merged])

            elif min_points > M:
                reconstructions.append([])

        return reconstructions, data, kps

    @staticmethod
    def choose_model(metrics_info, ckpt_paths, tail_frac=0.2):
        if "train_Lrc" not in metrics_info:
            raise KeyError("metrics_info must include 'train_Lrc'.")

        steps = np.asarray(metrics_info["train_Lrc"]["steps"], dtype=float)
        losses = np.asarray(metrics_info["train_Lrc"]["values"], dtype=float)
        assert steps.size == losses.size > 0, "Bad train_Lrc input"

        # Sort by step
        order = np.argsort(steps)
        steps, losses = steps[order], losses[order]

        # ---- Find global minimum ----
        j = int(np.nanargmin(losses))
        best_step = steps[j]
        best_val = losses[j]

        # ---- Map best training step → best checkpoint ----
        max_step = float(steps[-1])
        prog = best_step / max_step if max_step > 0 else 1.0

        ckpt_paths = TestBase.filter_and_sort_ckpts(ckpt_paths)
        N = len(ckpt_paths)

        idx = int(round(prog * (N - 1)))
        idx = min(max(idx, 0), N - 1)

        best_ckpt = ckpt_paths[idx][2]

        return {
            "ckpt": best_ckpt,
            "ckpt_index": idx,
            "total_ckpts": N,
            "best_step": float(best_step),
            "best_loss": float(best_val),
            "training_progress_%": round(prog * 100, 2),
            "picked_from": "global_min_chamfer_loss (step→epoch mapping)",
        }
