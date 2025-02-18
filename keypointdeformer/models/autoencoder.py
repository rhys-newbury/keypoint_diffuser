from torch.nn import Module

from .encoder_models.diffusion import DiffusionPoint, PointwiseNet, VarianceSchedule
from .encoder_models.encoders import PointTransformer


class AutoEncoder(Module):
    @staticmethod
    def modify_commandline_options(parser):
        return parser

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointTransformer(
            zdim=args.latent_dim, extra_latent=args.extra_latent
        )
        self.diffusion = DiffusionPoint(
            # +10
            net=PointwiseNet(
                point_dim=3,
                context_dim=args.latent_dim * 3 + args.extra_latent,
                residual=args.residual,
            ),
            var_sched=VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode,
            ),
        )

    def encode(self, x):
        """
        Args:
            x:  Point clouds to be encoded, (B, N, d).
        """
        code, _, sa = self.encoder(x)
        return code, sa

    def decode(self, code, num_points, flexibility=0.0, ret_traj=False):
        return self.diffusion.sample(
            num_points, code, flexibility=flexibility, ret_traj=ret_traj
        )

    def get_loss(self, x):
        code, sa = self.encode(x)
        loss = self.diffusion.get_loss(x, code)
        return loss, code, sa

    def forward(self, x):
        return self.encode(x)
