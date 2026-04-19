from dataclasses import dataclass
import torch
from torch import nn, autograd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class SAEConfig:
    d_model: int
    d_mid: int
    d_feat: int
    mid_l0: int
    feat_l0: int
    bandwidth: float
    batches_to_dead: int
    l0_coeff: float


@dataclass
class Results:
    loss: torch.Tensor
    l2_loss: torch.Tensor
    l0_loss: torch.Tensor
    n_dead0: int
    n_dead1: int
    n_dead2: int


class StepFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, thresh: torch.Tensor, bandwidth: float) -> torch.Tensor:
        ctx.save_for_backward(x, thresh)
        ctx.bandwidth = bandwidth

        return (x > thresh.exp()).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:  # type: ignore
        x, thresh = ctx.saved_tensors
        thresh = thresh.exp()
        bandwidth = ctx.bandwidth

        x_grad = torch.zeros_like(x)

        x_norm = (x - thresh) / bandwidth
        thresh_grad = (
            -(1.0 / bandwidth) * ((x_norm >= -0.5) & (x_norm <= 0.5)).float() * grad_output
        )

        return x_grad, thresh_grad, None


class JumpReLUFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, thresh: torch.Tensor, bandwidth: float) -> torch.Tensor:
        ctx.save_for_backward(x, thresh)
        ctx.bandwidth = bandwidth

        return x * (x > thresh.exp()).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:  # type: ignore
        x, thresh = ctx.saved_tensors
        thresh = thresh.exp()
        bandwidth = ctx.bandwidth

        x_grad = (x > thresh).float() * grad_output

        x_norm = (x - thresh) / bandwidth
        thresh_grad = (
            -(thresh / bandwidth) * ((x_norm >= -0.5) & (x_norm <= 0.5)).float() * grad_output
        )

        return x_grad, thresh_grad, None


class JumpReLU(nn.Module):
    def __init__(self, d_feat: int, bandwidth: float) -> None:
        super().__init__()
        self.thresh = nn.Parameter(torch.zeros((d_feat), device=device))
        self.bandwidth = bandwidth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return JumpReLUFunction.apply(x, self.thresh, self.bandwidth)


class DeepSAE(nn.Module):
    def __init__(self, cfg: SAEConfig) -> None:
        super().__init__()

        self.b_enc1 = nn.Parameter(torch.zeros(cfg.d_mid))
        self.b_enc2 = nn.Parameter(torch.zeros(cfg.d_feat))
        self.W_enc1 = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(cfg.d_model, cfg.d_mid)))
        self.W_enc2 = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(cfg.d_mid, cfg.d_feat)))

        self.b_dec2 = nn.Parameter(torch.zeros(cfg.d_mid))
        self.b_dec1 = nn.Parameter(torch.zeros(cfg.d_model))
        self.W_dec2 = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(cfg.d_feat, cfg.d_mid)))
        self.W_dec1 = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(cfg.d_mid, cfg.d_model)))

        self.W_dec1.data[:] = self.W_enc1.t().data
        self.W_dec2.data[:] = self.W_enc2.t().data

        self.jumprelu0 = JumpReLU(cfg.d_mid, cfg.bandwidth)
        self.jumprelu1 = JumpReLU(cfg.d_feat, cfg.bandwidth)
        self.jumprelu2 = JumpReLU(cfg.d_mid, cfg.bandwidth)

        self.bandwidth = cfg.bandwidth
        self.mid_l0 = cfg.mid_l0
        self.feat_l0 = cfg.feat_l0
        self.l0_coeff = cfg.l0_coeff

        self.batches_to_dead = cfg.batches_to_dead

        self.register_buffer("n_inactive_0", torch.zeros(cfg.d_mid))
        self.register_buffer("n_inactive_1", torch.zeros(cfg.d_feat))
        self.register_buffer("n_inactive_2", torch.zeros(cfg.d_mid))

    def _update_n_inactive(
        self, mid0: torch.Tensor, mid1: torch.Tensor, mid2: torch.Tensor
    ) -> None:
        for counter, act in zip(
            [self.n_inactive_0, self.n_inactive_1, self.n_inactive_2],
            [mid0, mid1, mid2],
            strict=True,
        ):
            counter += (act.sum(0) == 0).float()
            counter[act.sum(0) > 0] = 0

    def _l0_loss(
        self,
        mid0: torch.Tensor,
        mid1: torch.Tensor,
        mid2: torch.Tensor,
    ) -> torch.Tensor:
        l0_loss = torch.tensor(0.0, device=device)
        for act, thresh, k in zip(
            [mid0, mid1, mid2],
            [self.jumprelu0.thresh, self.jumprelu1.thresh, self.jumprelu2.thresh],
            [self.mid_l0, self.feat_l0, self.mid_l0],
            strict=True,
        ):
            l0_loss += self.l0_coeff * (
                (StepFunction.apply(act, thresh, self.bandwidth).sum(-1).mean() / k) - 1
            ).pow(2)

        return l0_loss

    def _loss_dict(
        self,
        input: torch.Tensor,
        recon: torch.Tensor,
        mid0: torch.Tensor,
        mid1: torch.Tensor,
        mid2: torch.Tensor,
    ) -> Results:
        l2_loss = (recon.float() - input.float()).pow(2).mean()
        l0_loss = self._l0_loss(mid0, mid1, mid2)
        loss = l2_loss + l0_loss

        return Results(
            loss=loss,
            l2_loss=l2_loss,
            l0_loss=l0_loss,
            n_dead0=int((self.n_inactive_0 > self.batches_to_dead).sum().item()),
            n_dead1=int((self.n_inactive_1 > self.batches_to_dead).sum().item()),
            n_dead2=int((self.n_inactive_2 > self.batches_to_dead).sum().item()),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Results]:
        input = x.float()
        mid0 = self.jumprelu0(input @ self.W_enc1 + self.b_enc1)
        mid1 = self.jumprelu1(mid0 @ self.W_enc2 + self.b_enc2)
        mid2 = self.jumprelu2(mid1 @ self.W_dec2 + self.b_dec2)
        recon = mid2 @ self.W_dec1 + self.b_dec1

        with torch.no_grad():
            self._update_n_inactive(mid0, mid1, mid2)

        return recon, self._loss_dict(input, recon, mid0, mid1, mid2)


class ShallowSAE(nn.Module):
    def __init__(self, cfg: SAEConfig) -> None:
        super().__init__()

        self.b_enc = nn.Parameter(torch.zeros(cfg.d_feat))
        self.W_enc = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(cfg.d_model, cfg.d_feat)))

        self.b_dec = nn.Parameter(torch.zeros(cfg.d_model))
        self.W_dec = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(cfg.d_feat, cfg.d_model)))

        self.W_dec.data[:] = self.W_enc.t().data

        self.jumprelu = JumpReLU(cfg.d_feat, cfg.bandwidth)

        self.batches_to_dead = cfg.batches_to_dead
        self.bandwidth = cfg.bandwidth
        self.feat_l0 = cfg.feat_l0
        self.l0_coeff = cfg.l0_coeff

        self.register_buffer("n_inactive", torch.zeros(cfg.d_feat))

    def _update_n_inactive(self, feat: torch.Tensor) -> None:
        self.n_inactive += (feat.sum(0) == 0).float()
        self.n_inactive[feat.sum(0) > 0] = 0

    def _loss_dict(
        self,
        input: torch.Tensor,
        recon: torch.Tensor,
        feat: torch.Tensor,
    ) -> Results:
        l2_loss = (recon.float() - input.float()).pow(2).mean()
        l0_loss = (
            (
                StepFunction.apply(feat, self.jumprelu.thresh, self.bandwidth).sum(-1).mean()
                / self.feat_l0
            )
            - 1
        ).pow(2)
        loss = l2_loss + l0_loss

        return Results(
            loss=loss,
            l2_loss=l2_loss,
            l0_loss=l0_loss,
            n_dead0=0,
            n_dead1=int((self.n_inactive > self.batches_to_dead).sum().item()),
            n_dead2=0,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Results]:
        input = x.float()
        feat = self.jumprelu(input @ self.W_enc + self.b_enc)
        recon = feat @ self.W_dec + self.b_dec

        with torch.no_grad():
            self._update_n_inactive(feat)

        return recon, self._loss_dict(input, recon, feat)
