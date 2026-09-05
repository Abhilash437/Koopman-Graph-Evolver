"""Shared multi-task loss weight defaults and helpers.

Paper defaults (Eq. total loss):
  L = 10.0 * L_recon + 1.0 * L_dyn + 2.0 * L_collapse + 5.0 * L_iso
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

DEFAULT_LOSS_WEIGHTS: Dict[str, float] = {
    "dyn": 1.0,
    "recon": 10.0,
    "collapse": 2.0,
    "iso": 5.0,
}


def resolve_loss_weights(
    lambda_dyn: Optional[float] = None,
    lambda_recon: Optional[float] = None,
    lambda_collapse: Optional[float] = None,
    lambda_iso: Optional[float] = None,
    overrides: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Build a loss-weight dict, falling back to paper defaults."""
    weights = dict(DEFAULT_LOSS_WEIGHTS)
    if overrides:
        weights.update({k: float(v) for k, v in overrides.items() if k in weights})
    if lambda_dyn is not None:
        weights["dyn"] = float(lambda_dyn)
    if lambda_recon is not None:
        weights["recon"] = float(lambda_recon)
    if lambda_collapse is not None:
        weights["collapse"] = float(lambda_collapse)
    if lambda_iso is not None:
        weights["iso"] = float(lambda_iso)
    return weights


def apply_loss_weights(model: Any, weights: Mapping[str, float]) -> None:
    """Attach loss weights to a model for use inside ``compute_loss``."""
    model.lambda_dyn = float(weights["dyn"])
    model.lambda_recon = float(weights["recon"])
    model.lambda_collapse = float(weights["collapse"])
    model.lambda_iso = float(weights["iso"])


def get_loss_weights(model: Any) -> Dict[str, float]:
    """Read loss weights from a model, with paper defaults as fallback."""
    return {
        "dyn": float(getattr(model, "lambda_dyn", DEFAULT_LOSS_WEIGHTS["dyn"])),
        "recon": float(getattr(model, "lambda_recon", DEFAULT_LOSS_WEIGHTS["recon"])),
        "collapse": float(getattr(model, "lambda_collapse", DEFAULT_LOSS_WEIGHTS["collapse"])),
        "iso": float(getattr(model, "lambda_iso", DEFAULT_LOSS_WEIGHTS["iso"])),
    }


def format_loss_weights(weights: Mapping[str, float]) -> str:
    return (
        f"dyn={weights['dyn']:g}, recon={weights['recon']:g}, "
        f"collapse={weights['collapse']:g}, iso={weights['iso']:g}"
    )
