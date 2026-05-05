import argparse
import torch
from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    LoggingConfig,
)
from .sae import DeepJumpReLUTrainingSAEConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a deep JumpReLU SAE")

    # SAE architecture
    parser.add_argument("--d-in", type=int, default=1152)
    parser.add_argument("--d-mid", type=int, default=4096)
    parser.add_argument("--d-sae", type=int, default=16384)

    # Model / data
    parser.add_argument("--model-name", type=str, default="gemma-3-1b-pt")
    parser.add_argument("--hook-name", type=str, default="blocks.6.hook_resid_post")
    parser.add_argument("--dataset-path", type=str, default="Skylion007/openwebtext")
    parser.add_argument(
        "--no-streaming", dest="streaming", action="store_false", default=True
    )

    # Training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-batch-size-tokens", type=int, default=4096)
    parser.add_argument("--context-size", type=int, default=256)
    parser.add_argument("--training-tokens", type=int, default=60000 * 4096)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--l0-coefficient", type=float, default=10.0)
    parser.add_argument(
        "--no-rescale",
        dest="rescale_acts_by_decoder_norm",
        action="store_false",
        default=True,
    )

    # Output
    parser.add_argument("--output-path", type=str, default="saes/saelens_run_1")

    args = parser.parse_args()

    cfg = LanguageModelSAERunnerConfig(
        sae=DeepJumpReLUTrainingSAEConfig(
            d_in=args.d_in,
            d_mid=args.d_mid,
            d_sae=args.d_sae,
            l0_coefficient=args.l0_coefficient,
            rescale_acts_by_decoder_norm=args.rescale_acts_by_decoder_norm,
            jumprelu_sparsity_loss_mode="tanh",
        ),
        model_name=args.model_name,
        hook_name=args.hook_name,
        dataset_path=args.dataset_path,
        streaming=args.streaming,
        lr=args.lr,
        train_batch_size_tokens=args.train_batch_size_tokens,
        context_size=args.context_size,
        training_tokens=args.training_tokens,
        device=args.device,
        logger=LoggingConfig(
            log_to_wandb=True,
            wandb_project="deep_sae",
            wandb_log_frequency=16,
            eval_every_n_wandb_logs=64,
        ),
    )

    sae = LanguageModelSAETrainingRunner(cfg).run()
    sae.save_inference_model(args.output_path)


if __name__ == "__main__":
    main()
