import torch
from torch import nn
from torch.nn import Module

from keypoint_diffuser.utils.loss import EDMLossCurriculum

from .diffusion import DiffusionPoint, PointwiseNet, PointwiseNetOld, VarianceSchedule
from .encoders.point_transformer import PointTransformerv2
from .vp_model import EDMPrecond


class AutoEncoder(Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointTransformerv2(
            zdim=args.latent_dim, extra_latent=args.extra_latent
        )

        self.fc_mu = nn.Sequential(
            nn.Linear(args.extra_latent, 64),
            nn.ReLU(),
            nn.Linear(64, args.extra_latent),
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(args.extra_latent, 64),
            nn.ReLU(),
            nn.Linear(64, args.extra_latent),
        )

        cls = PointwiseNetOld if args.use_old else PointwiseNet

        self.diffusion_ = cls(
            point_dim=3,
            context_dim=args.latent_dim * 3 + args.extra_latent,
            residual=args.residual,
        )
        self.use_edm = args.use_edm

        if self.use_edm:
            self.diffusion = EDMPrecond(self.diffusion_)
            self.loss = EDMLossCurriculum(max_steps=20000)
        else:
            self.diffusion = DiffusionPoint(
                net=self.diffusion_,
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
        code = self.encoder(x)

        z_kp = code[:, : self.args.latent_dim * 3]

        z_aux_raw = code[:, self.args.latent_dim * 3 :]

        mu = self.fc_mu(z_aux_raw)
        logvar = self.fc_logvar(z_aux_raw)

        return z_kp, mu, logvar

    def forward(self, x):
        return self.encode(x)

    def decode(self, code, num_points, flexibility=0.0, ret_traj=False):
        if self.use_edm:
            return self.diffusion.edm_sampler(code)

        return self.diffusion.sample(
            num_points, code, flexibility=flexibility, ret_traj=ret_traj
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_loss(self, x, step):
        z0, mu, logvar = self.encode(x)

        z_aux = self.reparameterize(mu, logvar)
        code = torch.cat([z0, z_aux], dim=1)

        t = x["target_shape"].view(-1, 5000, 3).cuda()
        if self.use_edm:
            loss = self.loss(
                net=self.diffusion, data=t, code=code.detach(), step=step
            ).mean()
        else:
            loss = self.diffusion.get_loss(t.transpose(1, 2), code)

        return loss, z0, mu, logvar
