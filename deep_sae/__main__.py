import argparse
import torch
from .train import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Deep JumpReLU SAE")
    parser.add_argument("--d-in", type=int, default=1152)
    parser.add_argument("--d-mid", type=int, default=4096)
    parser.add_argument("--d-sae", type=int, default=16384)
    parser.add_argument("--model-name", type=str, default="gemma-3-1b-pt")
    parser.add_argument("--hook-name", type=str, default="blocks.6.hook_resid_post")
    parser.add_argument("--dataset-path", type=str, default="Skylion007/openwebtext")
    parser.add_argument("--no-streaming", dest="streaming", action="store_false", default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-batch-size-tokens", type=int, default=4096)
    parser.add_argument("--context-size", type=int, default=256)
    parser.add_argument("--training-tokens", type=int, default=120_000 * 4096)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--l0-coefficient", type=float, default=15.0)
    parser.add_argument("--pre-act-loss-coefficient", type=float, default=4.0)
    parser.add_argument("--output-path", type=str, default="saes/run")
    parser.add_argument("--wandb-project", type=str, default="deep_sae")

    args = parser.parse_args()

    train(TrainConfig(
        d_in=args.d_in,
        d_mid=args.d_mid,
        d_sae=args.d_sae,
        model_name=args.model_name,
        hook_name=args.hook_name,
        dataset_path=args.dataset_path,
        streaming=args.streaming,
        lr=args.lr,
        train_batch_size_tokens=args.train_batch_size_tokens,
        context_size=args.context_size,
        training_tokens=args.training_tokens,
        device=args.device,
        dtype=args.dtype,
        l0_coefficient=args.l0_coefficient,
        pre_act_loss_coefficient=args.pre_act_loss_coefficient,
        output_path=args.output_path,
        wandb_project=args.wandb_project,
    ))


if __name__ == "__main__":
    main()
