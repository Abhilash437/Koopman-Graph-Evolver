import pytest
from koopman_evolver.cli import build_parser


def test_cli_train_parser():
    """Test CLI train argument parser flags."""
    parser = build_parser()
    args = parser.parse_args(["train", "--md17", "aspirin", "--model", "koopman", "--epochs", "50", "--seed", "42"])
    
    assert args.command == "train"
    assert args.md17 == "aspirin"
    assert args.model == "koopman"
    assert args.epochs == 50
    assert args.seed == 42


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
