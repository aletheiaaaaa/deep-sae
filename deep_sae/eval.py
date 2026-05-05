from collections.abc import Iterator
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from .sae import DeepJumpReLUSAE, DeepJumpReLUSAEConfig
from .utils import collect_acts


def _ce_loss(logits: torch.Tensor, tokens: torch.Tensor) -> float:
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        tokens[:, 1:].reshape(-1),
    ).item()


@torch.no_grad()
def eval_sae(
    model: HookedTransformer,
    sae: DeepJumpReLUSAE,
    sae_cfg: DeepJumpReLUSAEConfig,
    tok_iter: Iterator,
    n_eval_batches: int,
    hook_name: str,
    model_batch_size: int,
    device: str,
    dtype: torch.dtype,
) -> dict:
    """
    Compute SAELens-style evaluation metrics for a live model + SAE.
    Returns a flat dict of scalar metrics (no wandb calls).
    """
    d_in, d_sae = sae_cfg.d_in, sae_cfg.d_sae

    # Running stats for global explained variance:
    # EV = 1 - E[(x-x̂)²] / (E[x²] - E[x]²), per dim then averaged
    sum_x = torch.zeros(d_in, device=device, dtype=torch.float64)
    sum_x_sq = torch.zeros(d_in, device=device, dtype=torch.float64)
    sum_res_sq = torch.zeros(d_in, device=device, dtype=torch.float64)
    feature_fires = torch.zeros(d_sae, device=device, dtype=torch.long)

    l0_sum = l1_sum = 0.0
    l2_in_sq_sum = l2_out_sq_sum = l2_dot_sum = 0.0
    cos_sim_sum = 0.0
    n_tokens = 0

    all_token_batches: list[torch.Tensor] = []

    # Phase 1: reconstruction metrics
    for _ in range(n_eval_batches):
        seqs: list[list[int]] = []
        for _ in range(model_batch_size):
            try:
                seqs.append(next(tok_iter))
            except StopIteration:
                break
        if not seqs:
            break

        tokens = torch.tensor(seqs, device=device)
        all_token_batches.append(tokens)

        acts = collect_acts(model, seqs, hook_name, device, dtype)
        sae_out, feat_acts, _ = sae(acts)

        n = acts.shape[0]
        x64 = acts.double()
        xh64 = sae_out.double()
        res = x64 - xh64

        sum_x += x64.sum(0)
        sum_x_sq += x64.pow(2).sum(0)
        sum_res_sq += res.pow(2).sum(0)
        feature_fires += (feat_acts > 0).long().sum(0)

        l0_sum += (feat_acts > 0).float().sum(-1).sum().item()
        l1_sum += feat_acts.sum(-1).sum().item()

        l2_in = x64.norm(dim=-1)
        l2_out = xh64.norm(dim=-1)
        l2_in_sq_sum += l2_in.pow(2).sum().item()
        l2_out_sq_sum += l2_out.pow(2).sum().item()
        l2_dot_sum += (x64 * xh64).sum(-1).sum().item()
        cos_sim_sum += F.cosine_similarity(sae_out, acts, dim=-1).float().sum().item()
        n_tokens += n

    mean_x = sum_x / n_tokens
    E_x_sq = sum_x_sq / n_tokens
    total_var = (E_x_sq - mean_x.pow(2)).clamp(min=1e-12)
    explained_variance = (1.0 - (sum_res_sq / n_tokens) / total_var).mean().item()
    mse = (sum_res_sq / n_tokens).mean().item()
    l0 = l0_sum / n_tokens
    l1 = l1_sum / n_tokens
    l2_in_rms = (l2_in_sq_sum / n_tokens) ** 0.5
    l2_out_rms = (l2_out_sq_sum / n_tokens) ** 0.5
    l2_ratio = l2_out_rms / max(l2_in_rms, 1e-8)
    # Relative reconstruction bias (Eq. 10, https://arxiv.org/abs/2404.16014)
    relative_reconstruction_bias = (l2_out_sq_sum / n_tokens) / max(
        l2_dot_sum / n_tokens, 1e-8
    )
    cos_sim = cos_sim_sum / n_tokens
    feature_density = feature_fires.float() / n_tokens
    n_dead = int((feature_fires == 0).sum().item())
    frac_dead = n_dead / d_sae

    # Mean activation for mean-ablation CE baseline
    mean_act = mean_x.to(dtype).reshape(1, 1, -1)

    # Phase 2: CE loss
    ce_clean_list, ce_abl_list, ce_sae_list = [], [], []

    for tokens in all_token_batches:
        ce_clean_list.append(_ce_loss(model(tokens, return_type="logits"), tokens))

        def _ablate(value: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
            return mean_act.expand_as(value).to(value.dtype)

        ce_abl_list.append(
            _ce_loss(
                model.run_with_hooks(
                    tokens, fwd_hooks=[(hook_name, _ablate)], return_type="logits"
                ),
                tokens,
            )
        )

        def _sae_hook(value: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
            shape = value.shape
            flat = value.reshape(-1, shape[-1]).to(dtype)
            recon = sae.decode(sae.encode(flat)[0])
            return recon.reshape(shape).to(value.dtype)

        ce_sae_list.append(
            _ce_loss(
                model.run_with_hooks(
                    tokens, fwd_hooks=[(hook_name, _sae_hook)], return_type="logits"
                ),
                tokens,
            )
        )

    ce_clean = sum(ce_clean_list) / len(ce_clean_list)
    ce_ablated = sum(ce_abl_list) / len(ce_abl_list)
    ce_sae = sum(ce_sae_list) / len(ce_sae_list)
    ce_loss_score = (ce_ablated - ce_sae) / max(ce_ablated - ce_clean, 1e-8)

    # Weight-based metrics
    W_enc_mid = sae.W_enc_mid  # [d_mid, d_sae]
    W_dec_mid = sae.W_dec_mid  # [d_sae, d_mid]
    enc_norm = W_enc_mid.norm(dim=0)
    dec_norm = W_dec_mid.norm(dim=1)
    enc_dir = (W_enc_mid / enc_norm.clamp(min=1e-8)).T  # [d_sae, d_mid]
    dec_dir = W_dec_mid / dec_norm.unsqueeze(1).clamp(min=1e-8)
    enc_dec_cos = (dec_dir * enc_dir).sum(-1).mean().item()

    return {
        "explained_variance": explained_variance,
        "mse": mse,
        "cosine_similarity": cos_sim,
        "l0": l0,
        "l1": l1,
        "l2_norm_in": l2_in_rms,
        "l2_norm_out": l2_out_rms,
        "l2_ratio": l2_ratio,
        "relative_reconstruction_bias": relative_reconstruction_bias,
        "n_dead_features": n_dead,
        "frac_dead_features": frac_dead,
        "ce_loss_clean": ce_clean,
        "ce_loss_ablated": ce_ablated,
        "ce_loss_sae": ce_sae,
        "ce_loss_increase": ce_sae - ce_clean,
        "ce_loss_score": ce_loss_score,
        "mean_enc_dec_cosine_sim": enc_dec_cos,
        "mean_encoder_norm": enc_norm.mean().item(),
        # Feature density histogram values (list, caller logs however they want)
        "_feature_density": feature_density.cpu().float().tolist(),
    }
