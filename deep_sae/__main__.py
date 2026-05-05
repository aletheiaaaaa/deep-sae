import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    parser.add_argument(
        "--no-streaming", dest="streaming", action="store_false", default=True
    )
    parser.add_argument("--context-size", type=int, default=128)
    parser.add_argument("--model-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--train-batch-size-tokens", type=int, default=32768)
    parser.add_argument("--training-tokens", type=int, default=15_000 * 32768)
    parser.add_argument("--l0-coefficient", type=float, default=20.0)
    parser.add_argument("--pre-act-loss-coefficient", type=float, default=1e-6)
    parser.add_argument("--dead-neuron-window", type=int, default=125)
    parser.add_argument("--n-batches-in-buffer", type=int, default=32)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--output-path", type=str, default="saes/run")
    parser.add_argument("--wandb-project", type=str, default="deep_sae")
    parser.add_argument("--run-name", type=str, default=None, dest="wandb_run_name")
    parser.add_argument("--wandb-log-frequency", type=int, default=4)
    parser.add_argument("--wandb-hist-frequency", type=int, default=8)
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=8,
        help="Run full eval every N logging steps (0 = disabled)",
    )
    parser.add_argument(
        "--n-eval-batches",
        type=int,
        default=16,
        help="Number of model-batch-size batches to use per eval",
    )

    args = parser.parse_args()

    cfg = TrainConfig(
        d_in=args.d_in,
        d_mid=args.d_mid,
        d_sae=args.d_sae,
        bandwidth=2.0,
        jumprelu_tanh_scale=4.0,
        model_name=args.model_name,
        hook_name=args.hook_name,
        dataset_path=args.dataset_path,
        streaming=args.streaming,
        context_size=args.context_size,
        model_batch_size=args.model_batch_size,
        lr=args.lr,
        train_batch_size_tokens=args.train_batch_size_tokens,
        training_tokens=args.training_tokens,
        l0_coefficient=args.l0_coefficient,
        pre_act_loss_coefficient=args.pre_act_loss_coefficient,
        dead_neuron_window=args.dead_neuron_window,
        n_batches_in_buffer=args.n_batches_in_buffer,
        device=args.device,
        dtype=args.dtype,
        output_path=args.output_path,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_log_frequency=args.wandb_log_frequency,
        wandb_hist_frequency=args.wandb_hist_frequency,
        eval_frequency=args.eval_frequency,
        n_eval_batches=args.n_eval_batches,
    )

    train(cfg)


if __name__ == "__main__":
    main()
