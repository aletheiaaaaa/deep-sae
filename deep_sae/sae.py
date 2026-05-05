from dataclasses import dataclass
from typing import Any, override
from numpy.typing import NDArray
import torch
from torch import nn
from torch.nn import functional as F
from sae_lens import register_sae_training_class
from sae_lens.saes.sae import SAE, SAEConfig, TrainStepInput, TrainStepOutput
from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAEConfig, BatchTopKTrainingSAE
from sae_lens.saes.jumprelu_sae import JumpReLUSAEConfig


def act_times_W_dec(
    feature_acts: torch.Tensor,
    W_dec: torch.Tensor,
    rescale_acts_by_decoder_norm: bool,
) -> torch.Tensor:
    if rescale_acts_by_decoder_norm:
        feature_acts = feature_acts * (1 / W_dec.norm(dim=-1))
    return feature_acts @ W_dec


@dataclass
class DeepBTKTrainingSAEConfig(BatchTopKTrainingSAEConfig):
    """Configuration for deep BatchTopK SAE training."""

    d_mid: int = 4096  # type: ignore[assignment]
    rescale_acts_by_decoder_norm: bool = True
    decay_coefficient: float = 1e-3

    @override
    @classmethod
    def architecture(cls) -> str:
        return "deep_batchtopk"

    @override
    def get_inference_config_class(self) -> type[SAEConfig]:
        return DeepJumpReLUSAEConfig


class DeepBTKTrainingSAE(BatchTopKTrainingSAE):
    """Deep BatchTopK SAE for training.

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

    cfg: DeepBTKTrainingSAEConfig  # type: ignore[assignment]

    def __init__(self, cfg: DeepBTKTrainingSAEConfig, use_error_term: bool = False):
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

        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))

        return feature_acts, hidden_pre

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
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        feature_acts, hidden_pre = self.encode_with_hidden_pre(step_input.sae_in)
        sae_out = self.decode(feature_acts)

        per_item_mse_loss = self.mse_loss_fn(sae_out, step_input.sae_in)
        mse_loss = per_item_mse_loss.sum(dim=-1).mean()

        aux_losses = self.calculate_aux_loss(
            step_input=step_input,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            sae_out=sae_out,
        )

        total_loss = mse_loss + sum(loss_value for loss_value in aux_losses.values())

        losses = {"mse_loss": mse_loss}
        if isinstance(aux_losses, dict):
            losses.update(aux_losses)

        self.update_topk_threshold(feature_acts)

        return TrainStepOutput(
            sae_in=step_input.sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            loss=total_loss,
            losses=losses,
        )

    @override
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        aux_loss = self.calculate_topk_aux_loss(
            sae_in=step_input.sae_in,
            sae_out=sae_out,
            hidden_pre=hidden_pre,
            dead_neuron_mask=step_input.dead_neuron_mask,
        )

        decay_loss = self.cfg.decay_coefficient * feature_acts.pow(2).mean()

        return {"aux_loss": aux_loss, "decay_loss": decay_loss}

    @override
    def calculate_topk_aux_loss(
        self,
        sae_in: torch.Tensor,
        sae_out: torch.Tensor,
        hidden_pre: torch.Tensor,
        dead_neuron_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if dead_neuron_mask is None:
            return sae_out.new_tensor(0.0)

        num_dead = int(dead_neuron_mask.sum())

        if num_dead == 0:
            return sae_out.new_tensor(0.0)

        residual = (sae_in - sae_out).detach()
        k_aux = hidden_pre.shape[-1] // 2
        scale = min(num_dead / k_aux, 1.0)
        k_aux = min(k_aux, num_dead)

        auxk_latents = torch.where(dead_neuron_mask[None], hidden_pre, -torch.inf)
        auxk_topk = auxk_latents.topk(k_aux, sorted=False)
        auxk_acts = torch.zeros_like(hidden_pre).scatter(
            -1, auxk_topk.indices, auxk_topk.values
        )

        mid = act_times_W_dec(
            auxk_acts, self.W_dec_mid, self.cfg.rescale_acts_by_decoder_norm
        )
        recons = act_times_W_dec(
            F.relu(mid + self.b_dec_mid),
            self.W_dec_full,
            self.cfg.rescale_acts_by_decoder_norm,
        )

        auxk_loss = (recons - residual).pow(2).sum(dim=-1).mean()

        return self.cfg.aux_loss_coefficient * scale * auxk_loss

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
    "deep_batchtopk",
    DeepBTKTrainingSAE,
    DeepBTKTrainingSAEConfig,
)
