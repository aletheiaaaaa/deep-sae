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

    wandb.init(project="deep-sae", name="gemma_test_1")
    deep_optim = optim.AdamW(deep.parameters(), lr=train_cfg.lr)
    shallow_optim = optim.AdamW(shallow.parameters(), lr=train_cfg.lr)

    tokens_seen = 0

    for i, batch in enumerate(tqdm(loader)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        tokens_seen += int(attention_mask.sum().item())

        with model.trace(input_ids, attention_mask=attention_mask):
            hidden = model.model.layers[train_cfg.layer].output.save()

        _, dict_deep = deep(hidden)
        _, dict_shallow = shallow(hidden)

        deep_optim.zero_grad()
        dict_deep.l2_loss.backward()
        deep_optim.step()

        shallow_optim.zero_grad()
        dict_shallow.l2_loss.backward()
        shallow_optim.step()

        if (i + 1) % train_cfg.upload_every == 0:
            wandb.log(
                {
                    "l2_loss": dict_deep.l2_loss.item(),
                    "n_dead0": dict_deep.n_dead0,
                    "n_dead1": dict_deep.n_dead1,
                    "n_dead2": dict_deep.n_dead2,
                },
                step=i,
            )
            tqdm.write(f"Step {i + 1} | loss: {dict_deep.l2_loss.item():.4f} | ")

    os.makedirs(os.path.dirname(train_cfg.save_path), exist_ok=True)
    torch.save(deep.state_dict(), train_cfg.save_path)
    print(f"Saved model at {train_cfg.save_path} trained on {tokens_seen}")
