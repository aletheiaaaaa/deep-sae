from dataclasses import dataclass
import os
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb

from .model import DeepTopK


@dataclass
class TrainConfig:
    lr: float
    batch_size: int
    n_epochs: int
    frac_inactive: float
    save_path: str
    upload_every: int


@torch.no_grad()
def weights_topk(model: DeepTopK, frac_inactive: float) -> None:
    for param in model.parameters():
        flat = param.data.abs().flatten()
        k = int(flat.numel() * frac_inactive)
        thresh = flat.kthvalue(k).values
        mask = param.data.abs() > thresh
        param.data.mul_(mask)


def train(model: DeepTopK, acts: torch.Tensor, cfg: TrainConfig) -> None:
    dataset = TensorDataset(acts, acts)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)

    wandb.init(project="deep-sae")
    for epoch in tqdm(range(cfg.n_epochs)):
        frac_inactive = min(epoch * (2 * cfg.frac_inactive / cfg.n_epochs), cfg.frac_inactive)

        for inputs, _ in loader:
            _, loss_dict = model(inputs)

            optimizer.zero_grad()
            loss_dict.l2_loss.backward()
            optimizer.step()

            weights_topk(model, frac_inactive)

            if (epoch + 1) % cfg.upload_every == 0:
                wandb.log(
                    {
                        "l2_loss": loss_dict.l2_loss,
                        "l0_norm": loss_dict.l0_norm,
                        "n_dead": loss_dict.n_dead,
                    }
                )

        tqdm.write(f"Loss after {epoch + 1} epochs:", loss_dict.l2_loss)  # type: ignore

    if not os.path.exists(os.path.dirname(cfg.save_path)):
        os.mkdir(os.path.dirname(cfg.save_path))

    torch.save(model.state_dict(), cfg.save_path)
    print(f"Saved model at {cfg.save_path}")
