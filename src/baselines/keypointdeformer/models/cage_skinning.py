import pytorch3d.loss
import pytorch3d.utils
import torch
import torch.nn.parallel
import torch.utils.data
from einops import rearrange
from torch import nn

from ..utils.cages import deform_with_MVC
from ..utils.networks import Linear, MLPDeformer2, PointNetfeat
from ..utils.utils import normalize_to_box, sample_farthest_points


class CageSkinning(nn.Module):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument("--n_influence_ratio", type=float, help="", default=1.0)
        parser.add_argument("--lambda_init_points", type=float, help="", default=2.0)
        parser.add_argument("--lambda_chamfer", type=float, help="", default=1.0)
        parser.add_argument(
            "--lambda_influence_predict_l2", type=float, help="", default=1e6
        )
        parser.add_argument(
            "--iterations_init_points", type=float, help="", default=200
        )
        parser.add_argument("--no_optimize_cage", action="store_true", help="")
        parser.add_argument("--ico_sphere_div", type=int, help="", default=1)
        parser.add_argument("--n_fps", type=int, help="")
        return parser

    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        template_vertices, template_faces = self.create_cage()
        self.init_template(template_vertices, template_faces)
        self.init_networks(opt.bottleneck_size, 3, opt)
        self.init_optimizer()

    def create_cage(self):
        mesh = pytorch3d.utils.ico_sphere(self.opt.ico_sphere_div, device="cuda:0")
        init_cage_V = mesh.verts_padded()
        init_cage_F = mesh.faces_padded()
        init_cage_V = self.opt.cage_size * normalize_to_box(init_cage_V)[0]
        init_cage_V = init_cage_V.transpose(1, 2)
        return init_cage_V, init_cage_F

    def init_networks(self, bottleneck_size, dim, opt):
        # keypoint predictor
        shape_encoder_kpt = nn.Sequential(
            PointNetfeat(dim=dim, num_points=2048, bottleneck_size=bottleneck_size),
            Linear(
                bottleneck_size,
                bottleneck_size,
                activation="lrelu",
                normalization=opt.normalization,
            ),
        )
        nd_decoder_kpt = MLPDeformer2(
            dim=dim,
            bottleneck_size=bottleneck_size,
            npoint=opt.key_point,
            residual=opt.d_residual,
            normalization=opt.normalization,
        )
        self.keypoint_predictor = nn.Sequential(shape_encoder_kpt, nd_decoder_kpt)

        # influence predictor
        influence_size = self.opt.key_point * self.template_vertices.shape[2]
        shape_encoder_influence = nn.Sequential(
            PointNetfeat(dim=dim, num_points=2048, bottleneck_size=influence_size),
            Linear(
                influence_size,
                influence_size,
                activation="lrelu",
                normalization=opt.normalization,
            ),
        )
        dencoder_influence = nn.Sequential(
            Linear(
                influence_size,
                influence_size,
                activation="lrelu",
                normalization=opt.normalization,
            ),
            Linear(influence_size, influence_size, activation=None, normalization=None),
        )
        self.influence_predictor = nn.Sequential(
            shape_encoder_influence, dencoder_influence
        )

    def init_template(self, template_vertices, template_faces):
        # save template as buffer
        self.register_buffer("template_faces", template_faces)
        self.register_buffer("template_vertices", template_vertices)

        # key_point x number of vertices
        self.influence_param = nn.Parameter(
            torch.zeros(self.opt.key_point, self.template_vertices.shape[2]),
            requires_grad=True,
        )

    def init_optimizer(self):
        params = [{"params": self.influence_predictor.parameters()}]
        self.optimizer = torch.optim.Adam(params, lr=self.opt.lr)
        self.optimizer.add_param_group(
            {"params": self.influence_param, "lr": 10 * self.opt.lr}
        )
        params = [{"params": self.keypoint_predictor.parameters()}]
        self.keypoint_optimizer = torch.optim.Adam(params, lr=self.opt.lr)

    def optimize_cage(self, cage, shape, distance=0.4, iters=100, step=0.01):
        """
        pull cage vertices as close to the origin
        stop when distance to the shape is below the threshold
        """
        for _ in range(iters):
            vector = -cage
            current_distance = (
                torch.sum((cage[..., None] - shape[:, :, None]) ** 2, dim=1) ** 0.5
            )
            min_distance, _ = torch.min(current_distance, dim=2)
            do_update = min_distance > distance
            cage = cage + step * vector * do_update[:, None]
        return cage

    def forward(self, source_shape, target_shape):
        """
        source_shape (B,3,N)
        target_shape (B,3,M)
        """
        B, _, _ = source_shape.shape

        self.target_shape = target_shape

        if target_shape is not None:
            shape = torch.cat([source_shape, target_shape], dim=0)
        else:
            shape = source_shape

        keypoints = self.keypoint_predictor(shape)
        keypoints = torch.clamp(keypoints, -1.0, 1.0)
        if target_shape is not None:
            source_keypoints, target_keypoints = torch.split(keypoints, B, dim=0)
        else:
            source_keypoints = keypoints

        self.shape = shape
        self.keypoints = keypoints

        n_fps = self.opt.n_fps if self.opt.n_fps else 2 * self.opt.key_point
        self.init_keypoints = sample_farthest_points(shape, n_fps)

        if target_shape is not None:
            source_init_keypoints, target_init_keypoints = torch.split(
                self.init_keypoints, B, dim=0
            )
        else:
            source_init_keypoints = self.init_keypoints
            target_init_keypoints = None

        cage = self.template_vertices
        if not self.opt.no_optimize_cage:
            cage = self.optimize_cage(cage, source_shape)

        outputs = {
            "cage": cage.transpose(1, 2),
            "cage_face": self.template_faces,
            "source_keypoints": source_keypoints,
            "target_keypoints": target_keypoints,
            "source_init_keypoints": source_init_keypoints,
            "target_init_keypoints": target_init_keypoints,
        }

        self.influence = self.influence_param[None]
        self.influence_offset = self.influence_predictor(source_shape)
        self.influence_offset = rearrange(
            self.influence_offset,
            "b (k c) -> b k c",
            k=self.influence.shape[1],
            c=self.influence.shape[2],
        )
        self.influence = self.influence + self.influence_offset

        distance = torch.sum(
            (source_keypoints[..., None] - cage[:, :, None]) ** 2, dim=1
        )
        n_influence = int(
            (distance.shape[2] / distance.shape[1]) * self.opt.n_influence_ratio
        )
        n_influence = max(5, n_influence)
        threshold = torch.topk(distance, n_influence, largest=False)[0][:, :, -1]
        threshold = threshold[..., None]
        keep = distance <= threshold
        influence = self.influence * keep

        base_cage = cage
        keypoints_offset = target_keypoints - source_keypoints
        cage_offset = torch.sum(keypoints_offset[..., None] * influence[:, None], dim=2)
        new_cage = base_cage + cage_offset

        cage = cage.transpose(1, 2)
        new_cage = new_cage.transpose(1, 2)
        deformed_shapes, weights, _ = deform_with_MVC(
            cage,
            new_cage,
            self.template_faces.expand(B, -1, -1),
            source_shape.transpose(1, 2),
            verbose=True,
        )

        self.deformed_shapes = deformed_shapes

        outputs.update(
            {
                "cage": cage,
                "cage_face": self.template_faces,
                "new_cage": new_cage,
                "deformed": self.deformed_shapes,
                "weight": weights,
                "influence": influence,
            }
        )

        return outputs

    def compute_loss(self, iteration):
        losses = {}

        if self.opt.lambda_init_points > 0:
            init_points_loss = pytorch3d.loss.chamfer_distance(
                rearrange(self.keypoints, "b d n -> b n d"),
                rearrange(self.init_keypoints, "b d n -> b n d"),
            )[0]
            losses["init_points"] = self.opt.lambda_init_points * init_points_loss

        if self.opt.lambda_chamfer > 0:
            chamfer_loss = pytorch3d.loss.chamfer_distance(
                self.deformed_shapes, rearrange(self.target_shape, "b d n -> b n d")
            )[0]
            losses["chamfer"] = self.opt.lambda_chamfer * chamfer_loss

        if self.opt.lambda_influence_predict_l2 > 0:
            losses[
                "influence_predict_l2"
            ] = self.opt.lambda_influence_predict_l2 * torch.mean(
                self.influence_offset**2
            )

        return losses

    def _sum_losses(self, losses, names):
        return sum(v for k, v in losses.items() if k in names)

    def optimize(self, losses, iteration):
        self.keypoint_optimizer.zero_grad()
        self.optimizer.zero_grad()

        if iteration < self.opt.iterations_init_points:
            keypoints_loss = self._sum_losses(losses, ["init_points"])
            keypoints_loss.backward(retain_graph=True)
            self.keypoint_optimizer.step()

        if iteration >= self.opt.iterations_init_points:
            loss = self._sum_losses(
                losses, ["chamfer", "influence_predict_l2", "init_points"]
            )
            loss.backward()
            self.optimizer.step()
            self.keypoint_optimizer.step()

    def deform_from_keypoints(
        self,
        source_shape: torch.Tensor,  # (B, 3, N)
        target_keypoints: torch.Tensor,  # (B, 3, K) or (3, K)
        verbose: bool = True,
    ):
        """
        Deform `source_shape` so that its predicted keypoints move toward `target_keypoints`,
        using the cage + MVC pipeline in this module.

        Args:
            source_shape: (B, 3, N) point clouds in the same normalized space as training.
            target_keypoints: (B, 3, K) or (3, K) desired keypoint locations.
            n_influence_ratio: Optional override of opt.n_influence_ratio for how many
                cage verts each keypoint influences (relative to cage/keypoint counts).
            optimize_cage: If True, runs a quick shrink-to-shape step before deforming.
                If None, uses (not self.opt.no_optimize_cage).
            verbose: Passed through to deform_with_MVC.

        Returns:
            {
                "deformed": (B, N, 3)  # deformed points
                "cage": (B, Vc, 3),    # original cage (per batch)
                "new_cage": (B, Vc, 3),
                "weight": (B, N, Vc),  # MVC weights
                "source_keypoints": (B, 3, K),
                "target_keypoints": (B, 3, K),
                "influence": (B, K, Vc),  # masked influence used
            }
        """
        assert (
            source_shape.dim() == 3 and source_shape.shape[1] == 3
        ), "source_shape must be (B, 3, N)"
        B = source_shape.shape[0]
        device = source_shape.device
        dtype = source_shape.dtype

        # --- Normalize/prepare target keypoints batch shape ---
        if target_keypoints.dim() == 2:
            # (3, K) -> (B, 3, K)
            target_keypoints = target_keypoints.unsqueeze(0).expand(B, -1, -1)
        assert (
            target_keypoints.shape[0] == B and target_keypoints.shape[1] == 3
        ), "target_keypoints must be (B, 3, K) or (3, K)"

        K = target_keypoints.shape[2]

        # --- Predict source keypoints from the shape ---
        # (Optionally: keep these unclamped during inference as discussed.)
        source_keypoints = self.keypoint_predictor(source_shape)  # (B, 3, K)

        # --- Cage & faces on the right device ---
        cage = self.template_vertices.to(device=device, dtype=dtype)  # (1, 3, Vc)
        faces = self.template_faces.to(device)  # (1, Fc, 3)
        Vc = cage.shape[2]

        # Optional quick cage optimization
        optimize_cage = not getattr(self.opt, "no_optimize_cage", False)

        if optimize_cage:
            # run per-batch
            c = cage.expand(B, -1, -1).clone()
            with torch.no_grad():
                c = self.optimize_cage(c, source_shape)  # keeps (B, 3, Vc)
        else:
            c = cage.expand(B, -1, -1).clone()

        # --- Build influence matrix (keypoints x cage_verts), with predictor offset ---
        # Base learnable influence param
        base_influence = self.influence_param[None].to(
            device=device, dtype=dtype
        )  # (1, K, Vc)
        influence_offset = self.influence_predictor(source_shape)  # (B, K*Vc)
        influence_offset = rearrange(influence_offset, "b (k v) -> b k v", k=K, v=Vc)

        influence = base_influence + influence_offset  # (B, K, Vc)

        # --- Sparsify influences based on kpt↔cage proximity ---
        # distance between keypoints and cage verts
        # c: (B, 3, Vc), source_keypoints: (B, 3, K)
        dists_sq = torch.sum(
            (source_keypoints[..., None] - c[:, :, None]) ** 2, dim=1
        )  # (B, K, Vc)
        ratio = self.opt.n_influence_ratio
        n_influence = max(
            5, int((dists_sq.shape[2] / dists_sq.shape[1]) * float(ratio))
        )  # >=5

        # threshold per (B, K, 1) taking the n_influence nearest cage verts
        thresh = torch.topk(dists_sq, k=n_influence, largest=False, dim=2)[0][
            :, :, -1
        ].unsqueeze(-1)
        keep = dists_sq <= (
            thresh + 1e-12
        )  # (B, K, Vc) boolean mask, small epsilon to break ties
        influence = influence * keep

        # --- Move cage by aggregating keypoint offsets via influence ---
        kpt_offsets = target_keypoints - source_keypoints  # (B, 3, K)
        # weighted sum over K, broadcasting influence as (B, 1, K, Vc)
        cage_offset = torch.sum(
            kpt_offsets[:, :, :, None] * influence[:, None, :, :], dim=2
        )  # (B, 3, Vc)
        new_cage = c + cage_offset  # (B, 3, Vc)

        # --- MVC deformation ---
        # deform_with_MVC expects shapes as (B, V, 3)
        c_T = c.transpose(1, 2).contiguous()  # (B, Vc, 3)
        new_cage_T = new_cage.transpose(1, 2).contiguous()  # (B, Vc, 3)
        faces_B = faces.expand(B, -1, -1)  # (B, Fc, 3)
        src_T = source_shape.transpose(1, 2).contiguous()  # (B, N, 3)

        deformed_shapes, weights, _ = deform_with_MVC(
            c_T, new_cage_T, faces_B, src_T, verbose=verbose
        )  # deformed_shapes: (B, N, 3)

        return deformed_shapes
