from dataclasses import dataclass
import os
import torch
from torch import optim
from tqdm import tqdm
import wandb
import nnsight
from nnsight import LanguageModel
from datasets import load_dataset

from .model import DeepTopK

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainConfig:
    lr: float
    batch_size: int
    n_epochs: int
    frac_inactive: float
    save_path: str
    upload_every: int


@dataclass
class CacheConfig:
    model: str
    layer: int
    dataset: str
    batch_size: int


@torch.no_grad()
def weights_topk(model: DeepTopK, frac_inactive: float) -> None:
    for param in model.parameters():
        flat = param.data.abs().flatten()
        k = int(flat.numel() * frac_inactive)
        thresh = flat.kthvalue(k).values
        mask = param.data.abs() > thresh
        param.data.mul_(mask)


def train(sae: DeepTopK, cache_cfg: CacheConfig, cfg: TrainConfig) -> None:
    model = LanguageModel(cache_cfg.model, device_map="auto", dispatch=True)
    dataset = load_dataset(
        path=cache_cfg.dataset,
        split="train",
        streaming=True,
    )

    wandb.init(project="deep-sae")
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)

    batches = dataset.batch(batch_size=cache_cfg.batch_size)  # type: ignore
    for i, batch in enumerate(batches):
        frac_inactive = min(i * 1048576 / cache_cfg.batch_size, cfg.frac_inactive)
        with model.trace(batch):
            hidden = nnsight.save(model.transformer.h[cache_cfg.layer].outputs)

            _, loss_dict = sae(hidden)

            optimizer.zero_grad()
            loss_dict.l2_loss.backward()
            optimizer.step()

            weights_topk(model, frac_inactive)

            if (i + 1) % cfg.upload_every == 0:
                wandb.log(
                    {
                        "l2_loss": loss_dict.l2_loss,
                        "l0_norm": loss_dict.l0_norm,
                        "n_dead": loss_dict.n_dead,
                    }
                )

        tqdm.write(f"Loss after {i + 1} steps:", loss_dict.l2_loss)  # type: ignore

    if not os.path.exists(os.path.dirname(cfg.save_path)):
        os.mkdir(os.path.dirname(cfg.save_path))

    torch.save(model.state_dict(), cfg.save_path)
    print(f"Saved model at {cfg.save_path}")
