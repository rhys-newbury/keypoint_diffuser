from torch.nn import Module

from .diffusion import DiffusionPointOrig, PointwiseNetOld, VarianceSchedule
from .encoders.pointnet import PointNetEncoder


class AutoEncoderOrig(Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        context_dim = args.latent_dim * 3 + args.extra_latent
        self.encoder = PointNetEncoder(zdim=context_dim)
        self.diffusion = DiffusionPointOrig(
            net=PointwiseNetOld(
                point_dim=3, context_dim=context_dim, residual=args.residual
            ),
            var_sched=VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_t=args.beta_t,
                mode=args.sched_mode,
            ),
        )

    def encode(self, x):
        """
        Args:
            x:  Point clouds to be encoded, (B, N, d).
        """
        code, _ = self.encoder(x)
        return code

    def decode(self, code, num_points, flexibility=0.0, ret_traj=False):
        return self.diffusion.sample(
            num_points, code, flexibility=flexibility, ret_traj=ret_traj
        )

    def get_loss(self, x):
        code = self.encode(x)
        loss = self.diffusion.get_loss(x, code)
        return loss
