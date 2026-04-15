from dataclasses import dataclass
import os
import torch
from torch import optim
from tqdm import tqdm
import wandb
from nnsight import LanguageModel
from datasets import load_dataset

from .model import DeepTopK

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainConfig:
    lr: float
    batch_size: int
    frac_inactive: float
    save_path: str
    upload_every: int
    layer: int
    dataset: str


@torch.no_grad()
def weights_topk(model: DeepTopK, frac_inactive: float) -> None:
    for param in model.parameters():
        flat = param.data.abs().flatten()
        k = int(flat.numel() * frac_inactive)
        thresh = flat.kthvalue(k).values
        mask = param.data.abs() > thresh
        param.data.mul_(mask)


def train(sae: DeepTopK, train_cfg: TrainConfig) -> None:
    model = LanguageModel("google/gemma-3-1b-pt", device_map="auto", dispatch=True)
    dataset = load_dataset(
        path=train_cfg.dataset,
        split="train",
        streaming=True,
    )

    wandb.init(project="deep-sae")
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg.lr)

    batches = dataset.batch(batch_size=train_cfg.batch_size)  # type: ignore
    for i, batch in enumerate(batches):
        frac_inactive = min(i * 1048576 / train_cfg.batch_size, train_cfg.frac_inactive)
        with model.trace(**batch):
            hidden = model.model.layers[train_cfg.layer].output[0].save()

        _, loss_dict = sae(hidden.value)

        optimizer.zero_grad()
        loss_dict.l2_loss.backward()
        optimizer.step()

        weights_topk(sae, frac_inactive)

        if (i + 1) % train_cfg.upload_every == 0:
            wandb.log(
                {
                    "l2_loss": loss_dict.l2_loss,
                    "l0_norm": loss_dict.l0_norm,
                    "n_dead": loss_dict.n_dead,
                }
            )

        tqdm.write(f"Loss after {i + 1} steps:", loss_dict.l2_loss)  # type: ignore

    if not os.path.exists(os.path.dirname(train_cfg.save_path)):
        os.mkdir(os.path.dirname(train_cfg.save_path))

    torch.save(model.state_dict(), train_cfg.save_path)
    print(f"Saved model at {train_cfg.save_path}")
