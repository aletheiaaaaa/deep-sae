from dataclasses import dataclass
from pathlib import Path
import torch
import wandb
from datasets import load_dataset
from transformer_lens import HookedTransformer
from tqdm import tqdm

from .sae import DeepJumpReLUSAE, DeepJumpReLUSAEConfig
from .eval import eval_sae
from .utils import token_iter, collect_acts


@dataclass
class TrainConfig:
    # SAE architecture
    d_in: int = 1152
    d_mid: int = 4096
    d_sae: int = 16384
    bandwidth: float = 0.001
    jumprelu_tanh_scale: float = 1.0
    pre_act_loss_coefficient: float | None = 4.0

    # Model / data
    model_name: str = "gemma-3-1b-pt"
    hook_name: str = "blocks.6.hook_resid_post"
    dataset_path: str = "Skylion007/openwebtext"
    streaming: bool = True
    context_size: int = 256
    model_batch_size: int = 32  # prompts per model forward pass

    # Training
    lr: float = 1e-4
    train_batch_size_tokens: int = 4096
    training_tokens: int = 120_000 * 4096
    l0_coefficient: float = 15.0
    dead_neuron_window: int = 1000
    n_batches_in_buffer: int = 32

    # Output
    device: str = "cuda"
    dtype: str = "bfloat16"
    output_path: str = "saes/run"

    # Logging
    wandb_project: str = "deep_sae"
    wandb_log_frequency: int = 16  # training steps between metric logs
    wandb_hist_frequency: int = 1000  # training steps between density histogram logs
    eval_frequency: int = 0  # log steps between full evals (0 = disabled)
    n_eval_batches: int = 20  # model_batch_size-sized batches per eval


class ActivationBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.data: torch.Tensor | None = None

    def extend(self, acts: torch.Tensor) -> None:
        self.data = acts if self.data is None else torch.cat([self.data, acts])
        if len(self.data) > self.capacity:
            perm = torch.randperm(len(self.data), device=self.data.device)
            self.data = self.data[perm[: self.capacity]]

    def sample(self, n: int) -> torch.Tensor:
        assert self.data is not None and len(self.data) >= n
        perm = torch.randperm(len(self.data), device=self.data.device)
        batch, self.data = self.data[perm[:n]], self.data[perm[n:]]
        return batch

    def __len__(self) -> int:
        return 0 if self.data is None else len(self.data)


def train(cfg: TrainConfig) -> None:
    dtype = getattr(torch, cfg.dtype)
    device = cfg.device

    wandb.init(project=cfg.wandb_project, config=vars(cfg))

    model = HookedTransformer.from_pretrained(cfg.model_name, device=device)
    model.eval()
    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(cfg.dataset_path, split="train", streaming=cfg.streaming)
    train_tok_iter = token_iter(dataset, tokenizer, cfg.context_size)

    eval_tok_iter = None
    if cfg.eval_frequency > 0:
        eval_dataset = load_dataset(
            cfg.dataset_path, split="train", streaming=cfg.streaming
        )
        eval_tok_iter = token_iter(eval_dataset, tokenizer, cfg.context_size)

    sae_cfg = DeepJumpReLUSAEConfig(
        d_in=cfg.d_in,
        d_mid=cfg.d_mid,
        d_sae=cfg.d_sae,
        bandwidth=cfg.bandwidth,
        jumprelu_tanh_scale=cfg.jumprelu_tanh_scale,
        pre_act_loss_coefficient=cfg.pre_act_loss_coefficient,
    )
    sae = DeepJumpReLUSAE(sae_cfg, dtype=dtype, device=device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.lr, betas=(0.9, 0.999))

    buffer = ActivationBuffer(cfg.n_batches_in_buffer * cfg.train_batch_size_tokens)
    steps_since_fired = torch.zeros(cfg.d_sae, dtype=torch.long, device=device)
    hist_fire_counts = torch.zeros(cfg.d_sae, dtype=torch.long, device=device)

    def _refill() -> None:
        seqs: list[list[int]] = []
        for _ in range(cfg.model_batch_size):
            try:
                seqs.append(next(train_tok_iter))
            except StopIteration:
                break
        if seqs:
            buffer.extend(collect_acts(model, seqs, cfg.hook_name, device, dtype))

    # Prime buffer and init b_dec to activation mean
    _refill()
    if buffer.data is not None:
        with torch.no_grad():
            sae.b_dec.data = buffer.data.mean(0)

    total_steps = cfg.training_tokens // cfg.train_batch_size_tokens
    Path(cfg.output_path).mkdir(parents=True, exist_ok=True)

    for step in tqdm(range(1, total_steps + 1), desc="Training"):
        if len(buffer) < cfg.train_batch_size_tokens:
            _refill()

        batch = buffer.sample(cfg.train_batch_size_tokens)
        dead_neuron_mask = steps_since_fired > cfg.dead_neuron_window

        sae_out, feature_acts, hidden_pre = sae(batch)
        losses = sae.compute_loss(
            batch,
            sae_out,
            feature_acts,
            hidden_pre,
            l0_coefficient=cfg.l0_coefficient,
            dead_neuron_mask=dead_neuron_mask,
        )

        optimizer.zero_grad()
        losses["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0).item()
        optimizer.step()
        sae.normalize_decoders()

        fired_mask = feature_acts > 0
        fired = fired_mask.any(dim=0)
        steps_since_fired[fired] = 0
        steps_since_fired[~fired] += 1
        hist_fire_counts += fired_mask.long().sum(0)

        if step % cfg.wandb_log_frequency == 0:
            with torch.no_grad():
                l0 = fired_mask.float().sum(dim=-1).mean().item()
                residuals = sae_out - batch
                batch_mean = batch.mean(0)
                total_var = (batch - batch_mean).pow(2).mean(0).clamp(min=1e-12)
                explained_variance = (
                    (1.0 - residuals.pow(2).mean(0) / total_var).mean().item()
                )
                l2_ratio = (
                    (sae_out.norm(dim=-1) / batch.norm(dim=-1).clamp(min=1e-8))
                    .mean()
                    .item()
                )
                cos_sim = (
                    torch.nn.functional.cosine_similarity(sae_out, batch, dim=-1)
                    .mean()
                    .item()
                )
                frac_alive = (steps_since_fired == 0).float().mean().item()

            log: dict = {
                "loss": losses["loss"].item(),
                "recon_loss": losses["recon_loss"].item(),
                "l0_loss": losses["l0_loss"].item(),
                "l0": l0,
                "n_dead": dead_neuron_mask.sum().item(),
                "frac_alive": frac_alive,
                "tokens": step * cfg.train_batch_size_tokens,
                "explained_variance": explained_variance,
                "l2_ratio": l2_ratio,
                "cosine_similarity": cos_sim,
                "mean_threshold": sae.threshold.mean().item(),
                "grad_norm": grad_norm,
            }
            if "pre_act_loss" in losses:
                log["pre_act_loss"] = losses["pre_act_loss"].item()

            log_step = step // cfg.wandb_log_frequency
            if cfg.eval_frequency > 0 and log_step % cfg.eval_frequency == 0:
                assert eval_tok_iter is not None
                metrics = eval_sae(
                    model,
                    sae,
                    sae_cfg,
                    eval_tok_iter,
                    cfg.n_eval_batches,
                    cfg.hook_name,
                    cfg.model_batch_size,
                    device,
                    dtype,
                )
                density = metrics.pop("_feature_density")
                import numpy as np

                freq_np = np.array(density, dtype="float32")
                alive = freq_np > 0
                log.update({f"eval/{k}": v for k, v in metrics.items()})
                log["eval/feature_density_hist"] = wandb.Histogram(freq_np)
                if alive.any():
                    log["eval/log10_feature_density_hist"] = wandb.Histogram(
                        np.log10(freq_np[alive])
                    )

            wandb.log(log, step=step)

        if step % cfg.wandb_hist_frequency == 0:
            density = hist_fire_counts.float() / (
                cfg.wandb_hist_frequency * cfg.train_batch_size_tokens
            )
            freq_np = density.cpu().float().numpy()
            alive = freq_np > 0
            wandb.log(
                {
                    "feature_density_hist": wandb.Histogram(freq_np),
                    "log10_feature_density_hist": wandb.Histogram(
                        freq_np[alive] if alive.any() else freq_np[:1]
                    ),
                },
                step=step,
            )
            hist_fire_counts.zero_()

    torch.save(
        {"sae_state_dict": sae.state_dict(), "sae_cfg": sae_cfg, "train_cfg": cfg},
        Path(cfg.output_path) / "sae.pt",
    )
    wandb.finish()
