from dataclasses import dataclass
import os
import torch
import nnsight
from nnsight.modeling.vllm import VLLM
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class CacheConfig:
    model: str
    layer: int
    dataset: str
    batch_size: int
    save_dir: str


def make_act_cache(cfg: CacheConfig) -> torch.Tensor:
    model = VLLM(
        cfg.model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    dataset = load_dataset(
        path=cfg.dataset,
        split="train",
        streaming=True,
    )

    cache_list = []
    batches = dataset.batch(batch_size=cfg.batch_size)  # type: ignore
    for batch in batches:
        with model.trace(batch):
            hidden = nnsight.save(model.transformer.h[cfg.layer].outputs)
        cache_list.append(hidden)

    cache = torch.tensor(cache_list)
    if not os.path.exists(os.path.dirname(cfg.save_dir)):
        os.mkdir(os.path.dirname(cfg.save_dir))

    torch.save(cache, cfg.save_dir)
    print(f"Stored {cache.numel()} activations")

    return cache


def load_act_cache(cfg: CacheConfig) -> torch.Tensor:
    return torch.load(cfg.save_dir)
