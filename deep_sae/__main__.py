import argparse

from .train import TrainConfig, train
from .model import SAEConfig, DeepTopK, ShallowTopK, device


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Deep SAE experimentation",
        description="Trains and evaluates a deep vs shallow SAE on real-world transformers",
    )

    parser.add_argument("--layer", default=10, type=int)
    parser.add_argument("--dataset", default="Skylion007/openwebtext", type=str)

    parser.add_argument("--d_model", default=1152, type=int)
    parser.add_argument("--d_mid", default=2304, type=int)
    parser.add_argument("--d_feat", default=4608, type=int)
    parser.add_argument("--k_mid", default=144, type=int)
    parser.add_argument("--k_feat", default=72, type=int)
    parser.add_argument("--tokens_to_dead", default=10000000, type=int)

    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--upload_every", default=16, type=int)
    parser.add_argument("--save_path", default="./sae", type=str)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    train_cfg = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        save_path=args.save_path,
        upload_every=args.upload_every,
        layer=args.layer,
        dataset=args.dataset,
    )

    sae_cfg = SAEConfig(
        d_model=args.d_model,
        d_mid=args.d_mid,
        d_feat=args.d_feat,
        k_mid=args.k_mid,
        k_feat=args.k_feat,
        batches_to_dead=int(args.tokens_to_dead / (args.batch_size * 128)),
    )

    deep = DeepTopK(sae_cfg).to(device).float()
    shallow = ShallowTopK(sae_cfg).to(device).float()

    train(deep, shallow, train_cfg)


if __name__ == "__main__":
    main()
