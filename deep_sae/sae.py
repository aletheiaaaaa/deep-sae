from dataclasses import dataclass
from typing import Any, override
from numpy.typing import NDArray
import torch
from torch import nn
from torch.nn import functional as F
from sae_lens import register_sae_training_class
from sae_lens.saes.sae import SAE, SAEConfig, TrainStepInput, TrainStepOutput
from sae_lens.saes.jumprelu_sae import (
    JumpReLU,
    Step,
    JumpReLUSAEConfig,
    JumpReLUTrainingSAEConfig,
    JumpReLUTrainingSAE,
    calculate_pre_act_loss,
)


def act_times_W_dec(
    feature_acts: torch.Tensor,
    W_dec: torch.Tensor,
    rescale_acts_by_decoder_norm: bool,
) -> torch.Tensor:
    if rescale_acts_by_decoder_norm:
        feature_acts = feature_acts * (1 / W_dec.norm(dim=-1))
    return feature_acts @ W_dec


@dataclass
class DeepJumpReLUTrainingSAEConfig(JumpReLUTrainingSAEConfig):
    """Configuration for deep JumpReLU SAE training."""

    d_mid: int = 4096  # type: ignore[assignment]
    rescale_acts_by_decoder_norm: bool = True

    @override
    @classmethod
    def architecture(cls) -> str:
        return "deep_jumprelu"

    @override
    def get_inference_config_class(self) -> type[SAEConfig]:
        return DeepJumpReLUSAEConfig


class DeepJumpReLUTrainingSAE(JumpReLUTrainingSAE):
    """Deep JumpReLU SAE for training.

    Uses intermediate hidden layers in both the encoder and decoder.
    """

    W_enc_mid: nn.Parameter
    b_enc_mid: nn.Parameter
    W_enc_full: nn.Parameter
    b_enc_full: nn.Parameter

    W_dec_full: nn.Parameter
    b_dec_full: nn.Parameter
    W_dec_mid: nn.Parameter
    b_dec_mid: nn.Parameter

    cfg: DeepJumpReLUTrainingSAEConfig  # type: ignore[assignment]

    def __init__(self, cfg: DeepJumpReLUTrainingSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()

        self.b_enc_full = nn.Parameter(
            torch.zeros(self.cfg.d_mid, dtype=self.dtype, device=self.device)
        )
        self.b_enc_mid = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

        self.b_dec_mid = nn.Parameter(
            torch.zeros(self.cfg.d_mid, dtype=self.dtype, device=self.device)
        )
        self.b_dec_full = nn.Parameter(
            torch.zeros(self.cfg.d_in, dtype=self.dtype, device=self.device)
        )

        w_dec_mid_data = torch.empty(
            self.cfg.d_sae, self.cfg.d_mid, dtype=self.dtype, device=self.device
        )
        nn.init.kaiming_uniform_(w_dec_mid_data)
        w_dec_full_data = torch.empty(
            self.cfg.d_mid, self.cfg.d_in, dtype=self.dtype, device=self.device
        )
        nn.init.kaiming_uniform_(w_dec_full_data, mode="fan_out")

        self.W_dec_mid = nn.Parameter(w_dec_mid_data)
        self.W_dec_full = nn.Parameter(w_dec_full_data)

        self.W_enc_mid = nn.Parameter(self.W_dec_mid.data.T.clone().detach().contiguous())
        self.W_enc_full = nn.Parameter(
            self.W_dec_full.data.T.clone().detach().contiguous()
        )

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(
            F.relu(sae_in @ self.W_enc_full + self.b_enc_full) @ self.W_enc_mid
            + self.b_enc_mid
        )
        feature_acts = self.hook_sae_acts_post(
            JumpReLU.apply(hidden_pre, self.threshold, self.bandwidth)
        )
        return feature_acts, hidden_pre  # type: ignore[return-value]

    @override
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        mid = act_times_W_dec(
            feature_acts, self.W_dec_mid, self.cfg.rescale_acts_by_decoder_norm
        )
        sae_out_pre = (
            act_times_W_dec(
                F.relu(mid + self.b_dec_mid),
                self.W_dec_full,
                self.cfg.rescale_acts_by_decoder_norm,
            )
            + self.b_dec_full
        )
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    @override
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        threshold = self.threshold
        W_dec_norm = self.W_dec_mid.norm(dim=-1)

        if self.cfg.jumprelu_sparsity_loss_mode == "step":
            l0 = Step.apply(hidden_pre, threshold, self.bandwidth).sum(dim=-1)  # type: ignore[attr-defined]
            l0_loss = (step_input.coefficients["l0"] * l0).mean()
        elif self.cfg.jumprelu_sparsity_loss_mode == "tanh":
            per_item_l0_loss = torch.tanh(
                self.cfg.jumprelu_tanh_scale * feature_acts * W_dec_norm
            ).sum(dim=-1)
            l0_loss = (step_input.coefficients["l0"] * per_item_l0_loss).mean()
        else:
            raise ValueError(
                f"Invalid sparsity loss mode: {self.cfg.jumprelu_sparsity_loss_mode}"
            )

        losses: dict[str, torch.Tensor] = {"l0_loss": l0_loss}

        if self.cfg.pre_act_loss_coefficient is not None:
            losses["pre_act_loss"] = calculate_pre_act_loss(
                self.cfg.pre_act_loss_coefficient,
                threshold,
                hidden_pre,
                step_input.dead_neuron_mask,
                W_dec_norm,
            )

        return losses

    def normalize_decoders(self) -> None:
        with torch.no_grad():
            self.W_dec_mid.data /= self.W_dec_mid.data.norm(dim=-1, keepdim=True).clamp(
                min=1
            )
            self.W_dec_full.data /= self.W_dec_full.data.norm(dim=-1, keepdim=True).clamp(
                min=1
            )

    @torch.no_grad()
    @override
    def fold_W_dec_norm(self) -> None:
        pass  # normalization is handled by normalize_decoders; parent's W_dec is unused

    @override
    def log_histograms(self) -> dict[str, NDArray[Any]]:
        return {}


@dataclass
class DeepJumpReLUSAEConfig(JumpReLUSAEConfig):
    """Configuration class for a deep JumpReLU inference SAE."""

    d_mid: int = 4096  # type: ignore[assignment]
    rescale_acts_by_decoder_norm: bool = False

    @override
    @classmethod
    def architecture(cls) -> str:
        return "deep_jumprelu"


class DeepJumpReLUSAE(SAE[DeepJumpReLUSAEConfig]):
    W_enc_mid: nn.Parameter
    b_enc_mid: nn.Parameter
    W_enc_full: nn.Parameter
    b_enc_full: nn.Parameter

    W_dec_full: nn.Parameter
    b_dec_full: nn.Parameter
    W_dec_mid: nn.Parameter
    b_dec_mid: nn.Parameter

    threshold: nn.Parameter

    def __init__(self, cfg: DeepJumpReLUSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()

        self.b_enc_full = nn.Parameter(
            torch.zeros(self.cfg.d_mid, dtype=self.dtype, device=self.device)
        )
        self.b_enc_mid = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

        self.b_dec_mid = nn.Parameter(
            torch.zeros(self.cfg.d_mid, dtype=self.dtype, device=self.device)
        )
        self.b_dec_full = nn.Parameter(
            torch.zeros(self.cfg.d_in, dtype=self.dtype, device=self.device)
        )

        w_dec_mid_data = torch.empty(
            self.cfg.d_sae, self.cfg.d_mid, dtype=self.dtype, device=self.device
        )
        nn.init.kaiming_uniform_(w_dec_mid_data)
        w_dec_full_data = torch.empty(
            self.cfg.d_mid, self.cfg.d_in, dtype=self.dtype, device=self.device
        )
        nn.init.kaiming_uniform_(w_dec_full_data, mode="fan_out")

        self.threshold = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

        self.W_dec_mid = nn.Parameter(w_dec_mid_data)
        self.W_dec_full = nn.Parameter(w_dec_full_data)

        self.W_enc_mid = nn.Parameter(self.W_dec_mid.data.T.clone().detach().contiguous())
        self.W_enc_full = nn.Parameter(
            self.W_dec_full.data.T.clone().detach().contiguous()
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(
            F.relu(sae_in @ self.W_enc_full + self.b_enc_full) @ self.W_enc_mid
            + self.b_enc_mid
        )
        feature_acts = self.activation_fn(hidden_pre)
        jumprelu_mask = (hidden_pre > self.threshold).to(feature_acts.dtype)
        return self.hook_sae_acts_post(feature_acts * jumprelu_mask)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        mid = act_times_W_dec(
            feature_acts, self.W_dec_mid, self.cfg.rescale_acts_by_decoder_norm
        )
        sae_out_pre = (
            act_times_W_dec(
                F.relu(mid + self.b_dec_mid),
                self.W_dec_full,
                self.cfg.rescale_acts_by_decoder_norm,
            )
            + self.b_dec_full
        )
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)


register_sae_training_class(
    "deep_jumprelu",
    DeepJumpReLUTrainingSAE,
    DeepJumpReLUTrainingSAEConfig,
)
