import pytest
from koopman_evolver.cli import build_parser
from koopman_evolver.utils.loss_weights import DEFAULT_LOSS_WEIGHTS, resolve_loss_weights


def test_cli_train_parser():
    """Test CLI train argument parser flags."""
    parser = build_parser()
    args = parser.parse_args(["train", "--md17", "aspirin", "--model", "koopman", "--epochs", "50", "--seed", "42"])
    
    assert args.command == "train"
    assert args.md17 == "aspirin"
    assert args.model == "koopman"
    assert args.epochs == 50
    assert args.seed == 42
    assert args.lambda_collapse == DEFAULT_LOSS_WEIGHTS["collapse"]
    assert args.lambda_iso == DEFAULT_LOSS_WEIGHTS["iso"]


def test_cli_train_loss_weight_overrides():
    parser = build_parser()
    args = parser.parse_args([
        "train", "--md17", "aspirin", "--model", "gru",
        "--lambda-collapse", "0", "--lambda-iso", "0",
        "--run-tag", "noreg",
    ])
    assert args.lambda_collapse == 0.0
    assert args.lambda_iso == 0.0
    assert args.run_tag == "noreg"
    weights = resolve_loss_weights(
        lambda_dyn=args.lambda_dyn,
        lambda_recon=args.lambda_recon,
        lambda_collapse=args.lambda_collapse,
        lambda_iso=args.lambda_iso,
    )
    assert weights["collapse"] == 0.0
    assert weights["iso"] == 0.0
    assert weights["recon"] == 10.0


def test_cli_eval_parser():
    """Test CLI eval argument parser flags."""
    parser = build_parser()
    args = parser.parse_args([
        "eval",
        "--md17", "aspirin",
        "--koopman-ckpt", "checkpoints/koopman.pt",
        "--gru-ckpt", "checkpoints/gru.pt",
        "--rollout-steps", "29"
    ])
    
    assert args.command == "eval"
    assert args.md17 == "aspirin"
    assert args.koopman_ckpt == "checkpoints/koopman.pt"
    assert args.gru_ckpt == "checkpoints/gru.pt"
    assert args.rollout_steps == 29
