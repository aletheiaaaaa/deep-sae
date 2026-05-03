from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class SAEConfig:
    d_model: int
    d_mid: int
    d_feat: int
    k_feat: int
    k_aux: int
    batches_to_dead: int
    aux_coeff: float


@dataclass
class Results:
    loss: torch.Tensor
    l2_loss: torch.Tensor
    aux_loss: torch.Tensor
    n_dead: int


class DeepSAE(nn.Module):
    def __init__(self, cfg: SAEConfig) -> None:
        super().__init__()

        self.b_enc1 = nn.Parameter(torch.zeros(cfg.d_mid))
        self.b_enc2 = nn.Parameter(torch.zeros(cfg.d_feat))
        self.W_enc1 = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(cfg.d_mid, cfg.d_model)).t().contiguous()
        )
        self.W_enc2 = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(cfg.d_feat, cfg.d_mid)).t().contiguous()
        )

        self.b_dec2 = nn.Parameter(torch.zeros(cfg.d_mid))
        self.b_dec1 = nn.Parameter(torch.zeros(cfg.d_model))
        self.W_dec2 = nn.Parameter(torch.empty(cfg.d_feat, cfg.d_mid))
        self.W_dec1 = nn.Parameter(torch.empty(cfg.d_mid, cfg.d_model))

        self.W_dec1.data[:] = self.W_enc1.t().data
        self.W_dec2.data[:] = self.W_enc2.t().data

        self.batches_to_dead = cfg.batches_to_dead
        self.k_feat = cfg.k_feat
        self.k_aux = cfg.k_aux
        self.aux_coeff = cfg.aux_coeff

        self.register_buffer("n_inactive", torch.zeros(cfg.d_feat))

    def _update_n_inactive(self, feat: torch.Tensor) -> None:
        self.n_inactive += (feat.sum((0, 1)) == 0).float()
        self.n_inactive[feat.sum((0, 1)) > 0] = 0

    def _topk(self, x: torch.Tensor, k: int) -> torch.Tensor:
        topk = torch.topk(x.flatten(), k * x.shape[0] * x.shape[1], dim=-1)
        return (
            torch.zeros_like(x.flatten())
            .scatter(-1, topk.indices, topk.values)
            .reshape(x.shape)
        )

    def _aux_loss(
        self,
        input: torch.Tensor,
        recon: torch.Tensor,
        feat: torch.Tensor,
    ) -> torch.Tensor:
        aux_loss = torch.tensor(0, dtype=torch.float32, device=device)

        dead_features = self.n_inactive > self.batches_to_dead
        scale = min(dead_features.sum() / self.k_aux, 1.0)
        if dead_features.sum() > 0:
            residual = input - recon
            topk_aux = torch.topk(
                feat[:, :, dead_features],
                min(self.k_aux, int(dead_features.sum().item())),
                dim=-1,
            )
            acts_aux = torch.zeros_like(feat[:, :, dead_features]).scatter(
                -1, topk_aux.indices, topk_aux.values
            )
            recon_aux = (
                F.relu(acts_aux @ self.W_dec2[dead_features] + self.b_dec2) @ self.W_dec1
            )
            aux_loss = self.aux_coeff * scale * (recon_aux - residual).pow(2).mean()

        return aux_loss

    def _loss_dict(
        self,
        input: torch.Tensor,
        recon: torch.Tensor,
        feat: torch.Tensor,
    ) -> Results:
        l2_loss = (recon.float() - input.float()).pow(2).mean()
        aux_loss = self._aux_loss(input, recon, feat)
        loss = l2_loss + aux_loss

        return Results(
            loss=loss,
            l2_loss=l2_loss,
            aux_loss=aux_loss,
            n_dead=int((self.n_inactive > self.batches_to_dead).sum().item()),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Results]:
        x = x.float()
        input = x.clone().detach()

        mid0 = F.relu(input @ self.W_enc1 + self.b_enc1)

        pre1 = F.relu(mid0 @ self.W_enc2 + self.b_enc2)
        mid1 = self._topk(pre1, self.k_feat)

        mid2 = F.relu(mid1 @ self.W_dec2 + self.b_dec2)

        recon = mid2 @ self.W_dec1 + self.b_dec1

        with torch.no_grad():
            self._update_n_inactive(mid1)

        return recon, self._loss_dict(input, recon, pre1)


class ShallowSAE(nn.Module):
    def __init__(self, cfg: SAEConfig) -> None:
        super().__init__()

        self.b_enc = nn.Parameter(torch.zeros(cfg.d_feat))
        self.W_enc = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(cfg.d_model, cfg.d_feat))
        )

        self.b_dec = nn.Parameter(torch.zeros(cfg.d_model))
        self.W_dec = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(cfg.d_feat, cfg.d_model))
        )

        self.W_dec.data[:] = self.W_enc.t().data

        self.batches_to_dead = cfg.batches_to_dead
        self.k_feat = cfg.k_feat
        self.k_aux = cfg.k_aux
        self.aux_coeff = cfg.aux_coeff

        self.register_buffer("n_inactive", torch.zeros(cfg.d_feat))

    def _update_n_inactive(self, feat: torch.Tensor) -> None:
        self.n_inactive += (feat.sum((0, 1)) == 0).float()
        self.n_inactive[feat.sum((0, 1)) > 0] = 0

    def _topk(self, x: torch.Tensor, k: int) -> torch.Tensor:
        topk = torch.topk(x.flatten(), k * x.shape[0] * x.shape[1], dim=-1)
        return (
            torch.zeros_like(x.flatten())
            .scatter(-1, topk.indices, topk.values)
            .reshape(x.shape)
        )

    def _aux_loss(
        self,
        input: torch.Tensor,
        recon: torch.Tensor,
        feat: torch.Tensor,
    ) -> torch.Tensor:
        aux_loss = torch.tensor(0, dtype=torch.float32, device=device)

        dead_features = self.n_inactive > self.batches_to_dead
        scale = min(dead_features.sum() / self.k_aux, 1.0)
        if dead_features.sum() > 0:
            residual = input - recon
            topk_aux = torch.topk(
                feat[:, :, dead_features],
                min(self.k_aux, int(dead_features.sum().item())),
                dim=-1,
            )
            acts_aux = torch.zeros_like(feat[:, :, dead_features]).scatter(
                -1, topk_aux.indices, topk_aux.values
            )
            recon_aux = acts_aux @ self.W_dec[dead_features]
            aux_loss = self.aux_coeff * scale * (recon_aux - residual).pow(2).mean()

        return aux_loss

    def _loss_dict(
        self,
        input: torch.Tensor,
        recon: torch.Tensor,
        feat: torch.Tensor,
    ) -> Results:
        l2_loss = (recon.float() - input.float()).pow(2).mean()
        aux_loss = self._aux_loss(input, recon, feat)
        loss = l2_loss + aux_loss

        return Results(
            loss=loss,
            l2_loss=l2_loss,
            aux_loss=aux_loss,
            n_dead=int((self.n_inactive > self.batches_to_dead).sum().item()),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Results]:
        x = x.float()
        input = x.clone().detach()

        pre = F.relu(x @ self.W_enc + self.b_enc)
        x = self._topk(pre, self.k_feat)
        feat = x

        recon = x @ self.W_dec + self.b_dec

        with torch.no_grad():
            self._update_n_inactive(feat)

        return recon, self._loss_dict(input, recon, pre)
