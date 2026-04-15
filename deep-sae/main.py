import argparse
import os

from .train import TrainConfig, train
from .cache import CacheConfig, make_act_cache, load_act_cache
from .model import SAEConfig, DeepTopK


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Deep SAE experimentation",
        description="Trains and evaluates a deep vs shallow SAE on real-world transformers",
    )

    parser.add_argument("--cache_model", default="google/gemma-3-1b-pt", type=str)
    parser.add_argument("--cache_layer", default=10, type=int)
    parser.add_argument("--cache_dataset", default="Skylion007/openwebtext", type=str)
    parser.add_argument("--cache_batch_size", default=2048, type=int)
    parser.add_argument("--cache_save_dir", default="./acts", type=str)

    parser.add_argument("--d_model", default=1152, type=int)
    parser.add_argument("--d_mid", default=2304, type=int)
    parser.add_argument("--d_feat", default=4608, type=int)
    parser.add_argument("--k_mid", default=144, type=int)
    parser.add_argument("--k_feat", default=72, type=int)
    parser.add_argument("--batches_to_dead", default=1000000, type=int)

    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=4096, type=int)
    parser.add_argument("--n_epochs", default=10, type=int)
    parser.add_argument("--frac_inactive", default=0.5, type=float)
    parser.add_argument("--upload_every", default=1, type=int)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    train_cfg = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        frac_inactive=args.frac_inactive,
        save_path=args.save_path,
        upload_every=args.upload_every,
    )

    cache_cfg = CacheConfig(
        model=args.cache_model,
        layer=args.cache_layer,
        dataset=args.cache_dataset,
        batch_size=args.cache_batch_size,
        save_dir=args.cache_save_dir,
    )

    sae_cfg = SAEConfig(
        d_model=args.d_model,
        d_mid=args.d_mid,
        d_feat=args.d_feat,
        k_mid=args.k_mid,
        k_feat=args.k_feat,
        batches_to_dead=args.batches_to_dead,
    )

    if os.path.exists(args.cache_save_dir):
        cache = load_act_cache(cache_cfg)
    else:
        cache = make_act_cache(cache_cfg)

    sae = DeepTopK(sae_cfg)

    train(sae, cache, train_cfg)
