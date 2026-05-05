from dataclasses import dataclass
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F


def _rectangle(x: torch.Tensor) -> torch.Tensor:
    return ((x > -0.5) & (x < 0.5)).to(x)


class JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(
        x: torch.Tensor, threshold: torch.Tensor, bandwidth: float
    ) -> torch.Tensor:
        return (x * (x > threshold)).to(x)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple, output: torch.Tensor) -> None:
        x, threshold, bandwidth = inputs
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        # STE for all params: straight-through regardless of whether features fired
        x_grad = grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * _rectangle((x - threshold) / bandwidth)
            * grad_output
        ).sum(0)
        return x_grad, threshold_grad, None


class Step(torch.autograd.Function):
    @staticmethod
    def forward(
        x: torch.Tensor, threshold: torch.Tensor, bandwidth: float
    ) -> torch.Tensor:
        return (x > threshold).to(x)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple, output: torch.Tensor) -> None:
        x, threshold, bandwidth = inputs
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        threshold_grad = (
            -(1.0 / bandwidth) * _rectangle((x - threshold) / bandwidth) * grad_output
        ).sum(0)
        return grad_output, threshold_grad, None  # STE for x


@dataclass
class DeepJumpReLUSAEConfig:
    d_in: int
    d_mid: int
    d_sae: int
    bandwidth: float = 0.001
    jumprelu_tanh_scale: float = 4.0
    pre_act_loss_coefficient: float | None = None


class DeepJumpReLUSAE(nn.Module):
    def __init__(
        self,
        cfg: DeepJumpReLUSAEConfig,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        kw: dict = dict(dtype=dtype, device=device)

        self.b_dec = nn.Parameter(torch.zeros(cfg.d_in, **kw))
        self.b_enc_full = nn.Parameter(torch.zeros(cfg.d_mid, **kw))
        self.b_enc_mid = nn.Parameter(torch.zeros(cfg.d_sae, **kw))
        self.b_dec_mid = nn.Parameter(torch.zeros(cfg.d_mid, **kw))

        w_dec_mid = torch.empty(cfg.d_sae, cfg.d_mid, **kw)
        nn.init.kaiming_uniform_(w_dec_mid)
        w_dec_full = torch.empty(cfg.d_mid, cfg.d_in, **kw)
        nn.init.kaiming_uniform_(w_dec_full, mode="fan_out")

        self.W_dec_mid = nn.Parameter(w_dec_mid)
        self.W_dec_full = nn.Parameter(w_dec_full)
        self.W_enc_mid = nn.Parameter(self.W_dec_mid.data.T.clone().contiguous())
        self.W_enc_full = nn.Parameter(self.W_dec_full.data.T.clone().contiguous())
        # Log-threshold: actual threshold = exp(log_threshold), init t=0.1 → θ≈1.1
        self.log_threshold = nn.Parameter(torch.full((cfg.d_sae,), 0.1, **kw))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        threshold = self.log_threshold.exp()
        hidden_pre = (
            F.relu((x - self.b_dec) @ self.W_enc_full + self.b_enc_full) @ self.W_enc_mid
            + self.b_enc_mid
        )
        feature_acts = JumpReLU.apply(hidden_pre, threshold, self.cfg.bandwidth)
        return feature_acts, hidden_pre

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        mid = feature_acts @ self.W_dec_mid
        return F.relu(mid + self.b_dec_mid) @ self.W_dec_full + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_acts, hidden_pre = self.encode(x)
        return self.decode(feature_acts), feature_acts, hidden_pre

    def compute_loss(
        self,
        x: torch.Tensor,
        sae_out: torch.Tensor,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        l0_coefficient: float,
        dead_neuron_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        recon_loss = (sae_out - x).pow(2).sum(-1).mean()

        W_dec_norm = self.W_dec_mid.norm(dim=-1)
        l0 = torch.tanh(self.cfg.jumprelu_tanh_scale * feature_acts * W_dec_norm).sum(dim=-1)
        l0_loss = l0_coefficient * l0.mean()

        losses: dict[str, torch.Tensor] = {"recon_loss": recon_loss, "l0_loss": l0_loss}

        if self.cfg.pre_act_loss_coefficient is not None:
            threshold = self.log_threshold.exp()
            if dead_neuron_mask is None or not dead_neuron_mask.any():
                losses["pre_act_loss"] = hidden_pre.new_tensor(0.0)
            else:
                per_item = (
                    (threshold - hidden_pre).relu() * dead_neuron_mask * W_dec_norm
                ).sum(dim=-1)
                losses["pre_act_loss"] = self.cfg.pre_act_loss_coefficient * per_item.mean()

        losses["loss"] = sum(losses.values())
        return losses

    def normalize_decoders(self) -> None:
        with torch.no_grad():
            self.W_dec_mid.data /= self.W_dec_mid.data.norm(dim=-1, keepdim=True).clamp(min=1)
            self.W_dec_full.data /= self.W_dec_full.data.norm(dim=-1, keepdim=True).clamp(min=1)
