from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SAEConfig:
    d_model: int
    d_mid: int
    d_feat: int
    k_mid: int
    k_feat: int
    batches_to_dead: int


@dataclass
class Results:
    l2_loss: float
    l0_norm: int
    n_dead: int


class DeepTopK(nn.Module):
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

        self.batches_to_dead = cfg.batches_to_dead
        self.k_mid = cfg.k_mid
        self.k_feat = cfg.k_feat
        self.n_batches_inactive: torch.Tensor

    def _update_n_inactive(self, x: torch.Tensor) -> None:
        self.n_batches_inactive += (x.sum(0) == 0).float()
        self.n_batches_inactive[x.sum(0) > 0] = 0

    def _topk(self, x: torch.Tensor, k: int) -> torch.Tensor:
        topk = torch.topk(x.flatten(), k * x.shape[0])
        return torch.zeros_like(x.flatten()).scatter(-1, topk.indices, topk.values).reshape(x.shape)

    def _loss_dict(
        self,
        input: torch.Tensor,
        recon: torch.Tensor,
        topk: torch.Tensor,
    ) -> Results:
        l2_loss = (recon.float() - input.float()).pow(2).mean()
        l0_norm = (topk > 0).float().sum(-1).mean()
        n_dead = self.n_batches_inactive > self.batches_to_dead

        return Results(l2_loss=l2_loss.item(), l0_norm=int(l0_norm.item()), n_dead=n_dead)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Results]:
        input = x.clone()

        x = self._topk(F.relu(x @ self.W_enc1 + self.b_enc1), self.k_mid)
        x = self._topk(F.relu(x @ self.W_enc2 + self.b_enc2), self.k_feat)
        acts = x.clone()
        self._update_n_inactive(acts)

        x = self._topk(F.relu(x @ self.W_dec2 + self.b_dec2), self.k_mid)
        recon = x @ self.W_dec1 + self.b_dec1

        return recon, self._loss_dict(input, recon, acts)
