from dataclasses import dataclass
import os

import torch
from torch import optim
from tqdm import tqdm
import wandb
from nnsight import LanguageModel
from datasets import load_dataset
from torch.utils.data import DataLoader

from .model import DeepTopK, device


@dataclass
class TrainConfig:
    lr: float
    batch_size: int
    frac_inactive: float
    save_path: str
    upload_every: int
    layer: int
    dataset: str
    ramp_tokens: int = 16_777_216


@torch.no_grad()
def weights_topk(model: DeepTopK, frac_inactive: float) -> None:
    for param in model.parameters():
        flat = param.data.abs().flatten()
        n = flat.numel()
        k = int(n * frac_inactive)

        if k == 0:
            continue

        thresh = flat.kthvalue(k).values
        mask = (param.data.abs() > thresh).to(param.data.dtype)
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
        num_workers=0,
        pin_memory=True,
    )

    wandb.init(project="deep-sae")
    optimizer = optim.AdamW(sae.parameters(), lr=train_cfg.lr)

    tokens_seen = 0

    for i, batch in enumerate(tqdm(loader)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        tokens_seen += int(attention_mask.sum().item())

        frac_inactive = min(tokens_seen / train_cfg.ramp_tokens, 1.0) * train_cfg.frac_inactive

        with model.trace(input_ids, attention_mask=attention_mask):
            hidden = model.model.layers[train_cfg.layer].output.save()

        _, loss_dict = sae(hidden)

        optimizer.zero_grad()
        loss_dict.l2_loss.backward()
        optimizer.step()

        weights_topk(sae, frac_inactive)

        if (i + 1) % train_cfg.upload_every == 0:
            wandb.log(
                {
                    "l2_loss": loss_dict.l2_loss.item(),
                    "n_dead0": loss_dict.n_dead0,
                    "n_dead1": loss_dict.n_dead1,
                    "n_dead2": loss_dict.n_dead2,
                    "frac_inactive": frac_inactive,
                    "tokens_seen": tokens_seen,
                },
                step=i,
            )
            tqdm.write(
                f"Step {i + 1} | loss: {loss_dict.l2_loss.item():.4f} | "
                f"frac_inactive: {frac_inactive:.4f}"
            )

    os.makedirs(os.path.dirname(train_cfg.save_path), exist_ok=True)
    torch.save(sae.state_dict(), train_cfg.save_path)
    print(f"Saved model at {train_cfg.save_path} trained on {tokens_seen}")
