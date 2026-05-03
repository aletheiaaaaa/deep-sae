from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class SAEConfig:
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
    def __init__(self, cfg: SAEConfig, n_blocks: int, layer_sizes: list[int]) -> None:
        super().__init__()

        if len(layer_sizes) - 1 != n_blocks:
            raise ValueError(
                f"layer_sizes must have exactly n_blocks + 1 = {n_blocks + 1} elements, "
                f"got {len(layer_sizes)}"
            )

        n = n_blocks

        self.enc_W = nn.ParameterList()
        self.enc_b = nn.ParameterList()
        for i in range(n):
            wt = (
                nn.init.kaiming_uniform_(torch.empty(layer_sizes[i + 1], layer_sizes[i]))
                .t()
                .contiguous()
            )
            self.enc_W.append(nn.Parameter(wt))
            self.enc_b.append(nn.Parameter(torch.zeros(layer_sizes[i + 1])))

        self.dec_W = nn.ParameterList()
        self.dec_b = nn.ParameterList()
        for i in range(n):
            self.dec_W.append(
                nn.Parameter(torch.empty(layer_sizes[n - i], layer_sizes[n - i - 1]))
            )
            self.dec_b.append(nn.Parameter(torch.zeros(layer_sizes[n - i - 1])))

        for i in range(n):
            self.dec_W[i].data[:] = self.enc_W[n - 1 - i].t().data

        self.batches_to_dead = cfg.batches_to_dead
        self.k_feat = cfg.k_feat
        self.k_aux = cfg.k_aux
        self.aux_coeff = cfg.aux_coeff
        self.n_blocks = n

        self.register_buffer("n_inactive", torch.zeros(layer_sizes[-1]))

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
            h = acts_aux
            for i in range(self.n_blocks):
                wt = self.dec_W[0][dead_features] if i == 0 else self.dec_W[i]
                h = h @ wt
                if i < self.n_blocks - 1:
                    h = F.relu(h + self.dec_b[i])
            aux_loss = self.aux_coeff * scale * (h - residual).pow(2).mean()

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

    def forward(self, x):
        x = x.float()
        input = x.clone().detach()

        h = input
        for i in range(self.n_blocks):
            h = h @ self.enc_W[i] + self.enc_b[i]
            if i < self.n_blocks - 1:
                h = F.relu(h)
        pre = h
        feat = self._topk(F.relu(pre), self.k_feat)

        h = feat
        for i in range(self.n_blocks):
            h = h @ self.dec_W[i] + self.dec_b[i]
            if i < self.n_blocks - 1:
                h = F.relu(h)
        recon = h

        with torch.no_grad():
            self._update_n_inactive(feat)

        return recon, self._loss_dict(input, recon, pre)
