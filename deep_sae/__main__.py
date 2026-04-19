import argparse

from .train import TrainConfig, train
from .model import SAEConfig, DeepSAE, ShallowSAE, device


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Deep SAE experimentation",
        description="Trains and evaluates a deep vs shallow SAE on real-world transformers",
    )

    parser.add_argument("--layer", default=10, type=int)
    parser.add_argument("--dataset", default="Skylion007/openwebtext", type=str)

    parser.add_argument("--mid_expand", default=2, type=float)
    parser.add_argument("--feat_expand", default=4, type=float)
    parser.add_argument("--mid_l0", default=144, type=int)
    parser.add_argument("--feat_l0", default=72, type=int)
    parser.add_argument("--bandwidth", default=1e-3, type=float)
    parser.add_argument("--tokens_to_dead", default=1e7, type=int)

    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--upload_every", default=16, type=int)
    parser.add_argument("--save_path", default="sae", type=str)
    parser.add_argument("--run_name", default="jumprelu", type=str)
    parser.add_argument("--l0_coeff", default=1.0, type=float)

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
        run_name=args.run_name,
    )

    # TODO: clean this up when making public

    sae_cfg = SAEConfig(
        d_model=1152,
        d_mid=args.mid_expand * 1152,
        d_feat=args.feat_expand * 1152,
        mid_l0=args.mid_l0,
        feat_l0=args.feat_l0,
        bandwidth=args.bandwidth,
        batches_to_dead=int(args.tokens_to_dead / (args.batch_size * 128)),
        l0_coeff=args.l0_coeff,
    )

    deep = DeepSAE(sae_cfg).to(device).float()
    shallow = ShallowSAE(sae_cfg).to(device).float()

    train(deep, shallow, train_cfg)


if __name__ == "__main__":
    main()
