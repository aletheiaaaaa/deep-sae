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
    n_dead0: int
    n_dead1: int
    n_dead2: int


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
        self.n_inactive_layers = [
            torch.zeros(cfg.k_mid),
            torch.zeros(cfg.k_feat),
            torch.zeros(cfg.k_mid),
        ]

    def _update_n_inactive(
        self, mid0: torch.Tensor, mid1: torch.Tensor, mid2: torch.Tensor
    ) -> None:
        for counter, act in zip(self.n_inactive_layers, [mid0, mid1, mid2], strict=True):
            counter += (act.sum((0, 1)) == 0).float()
            counter[act.sum((0, 1)) > 0] = 0

    def _topk(self, x: torch.Tensor, k: int) -> torch.Tensor:
        print(x.shape)
        topk = torch.topk(x.flatten(), k * x.shape[0])
        return torch.zeros_like(x.flatten()).scatter(-1, topk.indices, topk.values).reshape(x.shape)

    def _loss_dict(
        self,
        input: torch.Tensor,
        recon: torch.Tensor,
    ) -> Results:
        l2_loss = (recon.float() - input.float()).pow(2).mean()
        n_dead = [(self.n_inactive_layers[i] > self.batches_to_dead).sum() for i in range(3)]

        return Results(
            l2_loss=l2_loss.item(),
            n_dead0=int(n_dead[0].item()),
            n_dead1=int(n_dead[1].item()),
            n_dead2=int(n_dead[2].item()),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Results]:
        input = x.clone().detach()

        x = self._topk(F.relu(x @ self.W_enc1 + self.b_enc1), self.k_mid)
        mid0 = x.clone().detach()

        x = self._topk(F.relu(x @ self.W_enc2 + self.b_enc2), self.k_feat)
        mid1 = x.clone()

        x = self._topk(F.relu(x @ self.W_dec2 + self.b_dec2), self.k_mid)
        mid2 = x.clone().detach()

        recon = x @ self.W_dec1 + self.b_dec1

        self._update_n_inactive(mid0, mid1, mid2)

        return recon, self._loss_dict(input, recon)
