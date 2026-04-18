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
    k_mid: int
    k_feat: int
    k_aux_mid: int
    k_aux_feat: int
    batches_to_dead: int
    aux_coeff: float


@dataclass
class Results:
    loss: torch.Tensor
    l2_loss: torch.Tensor
    aux_loss: torch.Tensor
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
        self.k_aux_mid = cfg.k_aux_mid
        self.k_aux_feat = cfg.k_aux_feat
        self.aux_coeff = cfg.aux_coeff

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
            counter += (act.sum((0, 1)) == 0).float()
            counter[act.sum((0, 1)) > 0] = 0

    def _topk(self, x: torch.Tensor, k: int) -> torch.Tensor:
        topk = torch.topk(x, k, dim=-1)
        return torch.zeros_like(x).scatter(-1, topk.indices, topk.values)

    def _partial_forward(self, x: torch.Tensor, dead_feats: torch.Tensor, idx: int):
        layers = [
            (self.W_enc2, self.b_enc2, self.k_feat),
            (self.W_dec2, self.b_dec2, self.k_mid),
            (self.W_dec1, self.b_dec1, None),
        ]
        for i, (W, b, k) in enumerate(layers):
            if i < idx:
                continue
            W_i = W[dead_feats] if i == idx else W
            x = x @ W_i + b
            if k is not None:
                x = self._topk(F.relu(x), k)
        return x

    def _aux_loss(
        self,
        input: torch.Tensor,
        recon: torch.Tensor,
        mid0: torch.Tensor,
        mid1: torch.Tensor,
        mid2: torch.Tensor,
    ) -> torch.Tensor:
        aux_loss = torch.tensor(0, dtype=torch.float32, device=device)
        for i, (buffer, acts, k_aux) in enumerate(
            zip(
                [self.n_inactive_0, self.n_inactive_1, self.n_inactive_2],
                [mid0, mid1, mid2],
                [self.k_aux_mid, self.k_aux_feat, self.k_aux_mid],
                strict=True,
            ),
        ):
            dead_features = buffer > self.batches_to_dead
            if dead_features.sum() > 0:
                residual = input - recon
                topk_aux = torch.topk(
                    acts[:, :, dead_features], min(k_aux, dead_features.sum()), dim=-1
                )
                acts_aux = torch.zeros_like(acts[:, :, dead_features]).scatter(
                    -1, topk_aux.indices, topk_aux.values
                )
                recon_aux = self._partial_forward(acts_aux, dead_features, i)
                aux_loss += (recon_aux - residual).pow(2).mean()

        return aux_loss

    def _loss_dict(
        self,
        input: torch.Tensor,
        recon: torch.Tensor,
        mid0: torch.Tensor,
        mid1: torch.Tensor,
        mid2: torch.Tensor,
    ) -> Results:
        l2_loss = (recon.float() - input.float()).pow(2).mean()
        aux_loss = self._aux_loss(input, recon, mid0, mid1, mid2)
        loss = l2_loss + self.aux_coeff * aux_loss

        return Results(
            loss=loss,
            l2_loss=l2_loss,
            aux_loss=aux_loss,
            n_dead0=int((self.n_inactive_0 > self.batches_to_dead).sum().item()),
            n_dead1=int((self.n_inactive_1 > self.batches_to_dead).sum().item()),
            n_dead2=int((self.n_inactive_2 > self.batches_to_dead).sum().item()),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Results]:
        x = x.float()
        input = x.clone().detach()

        pre0 = F.relu(x @ self.W_enc1 + self.b_enc1)
        x = self._topk(pre0, self.k_mid)
        mid0 = x

        pre1 = F.relu(x @ self.W_enc2 + self.b_enc2)
        x = self._topk(pre1, self.k_feat)
        mid1 = x

        pre2 = F.relu(x @ self.W_dec2 + self.b_dec2)
        x = self._topk(pre2, self.k_mid)
        mid2 = x

        recon = x @ self.W_dec1 + self.b_dec1

        with torch.no_grad():
            self._update_n_inactive(mid0, mid1, mid2)

        return recon, self._loss_dict(input, recon, pre0, pre1, pre2)


class ShallowTopK(nn.Module):
    def __init__(self, cfg: SAEConfig) -> None:
        super().__init__()

        self.b_enc = nn.Parameter(torch.zeros(cfg.d_feat))
        self.W_enc = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(cfg.d_model, cfg.d_feat)))

        self.b_dec = nn.Parameter(torch.zeros(cfg.d_model))
        self.W_dec = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(cfg.d_feat, cfg.d_model)))

        self.batches_to_dead = cfg.batches_to_dead
        self.k_feat = cfg.k_feat
        self.k_aux_feat = cfg.k_aux_feat
        self.aux_coeff = cfg.aux_coeff

        self.register_buffer("n_inactive", torch.zeros(cfg.d_feat))

    def _update_n_inactive(self, feat: torch.Tensor) -> None:
        self.n_inactive += (feat.sum((0, 1)) == 0).float()
        self.n_inactive[feat.sum((0, 1)) > 0] = 0

    def _topk(self, x: torch.Tensor, k: int) -> torch.Tensor:
        topk = torch.topk(x, k, dim=-1)
        return torch.zeros_like(x).scatter(-1, topk.indices, topk.values)

    def _aux_loss(
        self,
        input: torch.Tensor,
        recon: torch.Tensor,
        feat: torch.Tensor,
    ) -> torch.Tensor:
        aux_loss = torch.tensor(0, dtype=torch.float32, device=device)

        dead_features = self.n_inactive > self.batches_to_dead
        if dead_features.sum() > 0:
            residual = input - recon
            k_aux = min(self.k_aux_feat, dead_features.sum())
            topk_aux = torch.topk(feat[:, :, dead_features], k_aux, dim=-1)
            acts_aux = torch.zeros_like(feat[:, :, dead_features]).scatter(
                -1, topk_aux.indices, topk_aux.values
            )
            recon_aux = acts_aux @ self.W_dec[dead_features] + self.b_dec
            aux_loss = (recon_aux - residual).pow(2).mean()

        return aux_loss

    def _loss_dict(
        self,
        input: torch.Tensor,
        recon: torch.Tensor,
        feat: torch.Tensor,
    ) -> Results:
        l2_loss = (recon.float() - input.float()).pow(2).mean()
        aux_loss = self._aux_loss(input, recon, feat)
        loss = l2_loss + self.aux_coeff * aux_loss

        return Results(
            loss=loss,
            l2_loss=l2_loss,
            aux_loss=aux_loss,
            n_dead0=0,
            n_dead1=int((self.n_inactive > self.batches_to_dead).sum().item()),
            n_dead2=0,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Results]:
        x = x.float()
        input = x.clone().detach()

        x = self._topk(F.relu(x @ self.W_enc + self.b_enc), self.k_feat)
        feat = x.clone().detach()

        recon = x @ self.W_dec + self.b_dec

        self._update_n_inactive(feat)

        return recon, self._loss_dict(input, recon, feat)
