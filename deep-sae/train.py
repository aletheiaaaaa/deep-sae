from dataclasses import dataclass
import os
import torch
from torch import optim
from tqdm import tqdm
import wandb
from nnsight import LanguageModel
from datasets import load_dataset
from torch.utils.data import DataLoader

from .model import DeepTopK

device = torch.device("cuda:0")


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
    model = LanguageModel("google/gemma-3-1b-pt", device_map=device, torch_dtype=torch.float16)
    tokenizer = model.tokenizer

    dataset = load_dataset(
        path=train_cfg.dataset,
        split="train",
        streaming=True,
    )

    def tokenize(examples):
        enc = tokenizer(
            examples["text"],
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    loader = DataLoader(
        dataset.with_format("torch"),
        batch_size=train_cfg.batch_size,
        num_workers=4,
        pin_memory=True,
    )

    wandb.init(project="deep-sae")
    optimizer = optim.AdamW(sae.parameters(), lr=train_cfg.lr)

    for i, batch in enumerate(tqdm(loader)):
        frac_inactive = min(i * 1048576 / train_cfg.batch_size, train_cfg.frac_inactive)

        input_ids = torch.tensor(batch["input_ids"], device=device)
        attention_mask = torch.tensor(batch["attention_mask"], device=device)

        with model.trace(input_ids, attention_mask=attention_mask):
            hidden = model.model.layers[train_cfg.layer].output[0].save()

        _, loss_dict = sae(hidden.value)

        optimizer.zero_grad()
        loss_dict.l2_loss.backward()
        optimizer.step()

        weights_topk(sae, frac_inactive)

        if (i + 1) % train_cfg.upload_every == 0:
            wandb.log(
                {
                    "l2_loss": loss_dict.l2_loss.item(),
                    "l0_norm": loss_dict.l0_norm.item(),
                    "n_dead": loss_dict.n_dead,
                }
            )

        tqdm.write(f"Loss after {i + 1} steps: {loss_dict.l2_loss.item()}")

    os.makedirs(os.path.dirname(train_cfg.save_path), exist_ok=True)
    torch.save(sae.state_dict(), train_cfg.save_path)
    print(f"Saved model at {train_cfg.save_path}")
