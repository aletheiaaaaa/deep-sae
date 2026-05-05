from collections.abc import Iterator
import torch
from transformer_lens import HookedTransformer


def token_iter(dataset, tokenizer, context_size: int) -> Iterator[list[int]]:
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    buf: list[int] = []
    for example in dataset:
        buf.extend(bos + tokenizer.encode(example["text"]))
        while len(buf) >= context_size:
            yield buf[:context_size]
            buf = buf[context_size:]


@torch.no_grad()
def collect_acts(
    model: HookedTransformer,
    seqs: list[list[int]],
    hook_name: str,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    tokens = torch.tensor(seqs, device=device)
    _, cache = model.run_with_cache(tokens, names_filter=hook_name, return_type=None)
    acts = cache[hook_name]  # [batch, seq, d_in]
    return acts.reshape(-1, acts.shape[-1]).to(dtype)
