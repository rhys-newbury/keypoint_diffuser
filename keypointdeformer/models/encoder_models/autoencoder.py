from torch.nn import Module

from .diffusion import DiffusionPoint, PointwiseNet, VarianceSchedule
from .encoders import PointTransformerv2
from .vp_model import VPPrecond
from keypointdeformer.utils.loss import EDMLoss

class AutoEncoder(Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointTransformerv2(
            zdim=args.latent_dim, extra_latent=args.extra_latent
        )
        self.diffusion_ = DiffusionPoint(
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
        self.diffusion = VPPrecond(self.diffusion_)
        self.loss = EDMLoss()

    def encode(self, x):
        """
        Args:
            x:  Point clouds to be encoded, (B, N, d).
        """
        code = self.encoder(x)
        return code

    def decode(self, code, num_points, flexibility=0.0, ret_traj=False):
        return self.diffusion.sample(
            num_points, code, flexibility=flexibility, ret_traj=ret_traj
        )

    def get_loss(self, x, use_perceptual_loss=False):
        code = self.encode(x)
        t = x["target_shape"].view(-1, 5000, 3).cuda()
        loss = self.loss(net=self.diffusion, data=t, code=code).sum()

        # loss = self.diffusion.get_loss(
        #     t.transpose(1, 2), code, use_perceptual_loss=use_perceptual_loss
        # )
        return loss, code
