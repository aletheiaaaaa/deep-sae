import argparse

from .train import TrainConfig, train_sae
from .sae import SAEConfig, DeepSAE, device


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Deep SAE experimentation",
        description="Trains and evaluates a deep vs shallow SAE on real-world "
        "transformers",
    )

    parser.add_argument("--layer", default=10, type=int)
    parser.add_argument("--dataset", default="Skylion007/openwebtext", type=str)

    parser.add_argument("--d_model", default=1152, type=int)
    parser.add_argument("--d_mid", default=2304, type=int)
    parser.add_argument("--d_feat", default=4608, type=int)
    parser.add_argument("--k_feat", default=16, type=int)
    parser.add_argument("--k_aux", default=512, type=int)
    parser.add_argument("--tokens_to_dead", default=1000000, type=int)

    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--upload_every", default=16, type=int)
    parser.add_argument("--save_path", default="sae", type=str)
    parser.add_argument("--run_name", default="batchtopk_fixed", type=str)
    parser.add_argument("--aux_coeff", default=6.0, type=float)

    return parser.parse_args()


def main() -> None:
    args = get_args()

    train_cfg = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        save_path=args.save_path,
        upload_every=args.upload_every,
        layer=args.layer,
        dataset=args.dataset,
        run_name=args.run_name,
    )

    sae_cfg = SAEConfig(
        k_feat=args.k_feat,
        k_aux=args.k_aux,
        batches_to_dead=int(args.tokens_to_dead / (args.batch_size * 128)),
        aux_coeff=args.aux_coeff,
    )

    deep = (
        DeepSAE(sae_cfg, n_blocks=2, layer_sizes=[args.d_model, args.d_mid, args.d_feat])
        .to(device)
        .float()
    )
    shallow = (
        DeepSAE(sae_cfg, n_blocks=1, layer_sizes=[args.d_model, args.d_feat])
        .to(device)
        .float()
    )

    train_sae(deep, shallow, train_cfg)


if __name__ == "__main__":
    main()
