from dataclasses import dataclass
import os

import torch
from torch import optim
from tqdm import tqdm
import wandb
from nnsight import LanguageModel
from datasets import load_dataset
from torch.utils.data import DataLoader

from .model import DeepTopK, ShallowTopK, device


@dataclass
class TrainConfig:
    lr: float
    batch_size: int
    save_path: str
    upload_every: int
    layer: int
    dataset: str
    run_name: str


def train(deep: DeepTopK, shallow: ShallowTopK, train_cfg: TrainConfig) -> None:
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

    wandb.init(project="deep-sae", name=train_cfg.run_name)
    deep_optim = optim.AdamW(deep.parameters(), lr=train_cfg.lr)
    shallow_optim = optim.AdamW(shallow.parameters(), lr=train_cfg.lr)

    tokens_seen = 0

    for i, batch in enumerate(tqdm(loader)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        tokens_seen += int(attention_mask.sum().item())

        with model.trace(input_ids, attention_mask=attention_mask):
            hidden = model.model.layers[train_cfg.layer].output.save()

        h = hidden.detach()
        _, dict_deep = deep(h.clone())
        _, dict_shallow = shallow(h.clone())

        deep_optim.zero_grad()
        dict_deep.l2_loss.backward()
        deep_optim.step()

        shallow_optim.zero_grad()
        dict_shallow.l2_loss.backward()
        shallow_optim.step()

        if (i + 1) % train_cfg.upload_every == 0:
            wandb.log(
                {
                    "deep/l2_loss": dict_deep.l2_loss.item(),
                    "deep/n_dead0": dict_deep.n_dead0,
                    "deep/n_dead1": dict_deep.n_dead1,
                    "deep/n_dead2": dict_deep.n_dead2,
                    "shallow/l2_loss": dict_shallow.l2_loss.item(),
                    "shallow/n_dead1": dict_shallow.n_dead1,
                },
                step=i,
            )
            tqdm.write(f"Step {i + 1} | loss: {dict_deep.l2_loss.item():.4f} | ")

    os.makedirs(train_cfg.save_path, exist_ok=True)

    save_path = os.path.join(train_cfg.save_path, train_cfg.run_name)

    torch.save(deep.state_dict(), f"{save_path}_deep.pt")
    torch.save(shallow.state_dict(), f"{save_path}_shallow.pt")
    print(f"Saved SAEs at {save_path}_{{deep/shallow}}.pt trained on {tokens_seen}")
