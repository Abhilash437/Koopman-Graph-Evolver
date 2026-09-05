#!/bin/bash
# ==============================================================================
# Eval-only: re-run aspirin rebuttal evaluations from existing checkpoints.
# Does NOT retrain. Writes a text log with Phase-9 dual ratios (R_norm + R_edge).
#
# Usage (on GCP, from repo root; log is written automatically):
#   chmod +x run_rebuttal_aspirin_eval_only.sh
#   ./run_rebuttal_aspirin_eval_only.sh
#
# Env knobs:
#   LOG_FILE=eval_logs/rebuttal_1LAu_aspirin.log
# ==============================================================================

set -euo pipefail

SEEDS="${SEEDS:-42 1337 2026}"
DEVICE="${DEVICE:-cuda}"
CKPT_DIR="${CKPT_DIR:-./checkpoints/rebuttal_1LAu}"
OUT_DIR="${OUT_DIR:-./results/rebuttal_1LAu/aspirin}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-29}"
LOG_FILE="${LOG_FILE:-eval_logs/rebuttal_1LAu_aspirin.log}"
PYTHON="${PYTHON:-python3}"
export PYTHONUNBUFFERED=1
CLI=(${PYTHON} -u -m koopman_evolver.cli)

# shellcheck disable=SC2206
SEED_LIST=(${SEEDS})
TAGS=(full noreg)

mkdir -p "${OUT_DIR}" "$(dirname "${LOG_FILE}")"

LOG_FILE="$(cd "$(dirname "${LOG_FILE}")" && pwd)/$(basename "${LOG_FILE}")"
if [[ -z "${_REBUTTAL_LOGGING:-}" ]]; then
  export _REBUTTAL_LOGGING=1
  : > "${LOG_FILE}"
  exec > >(tee -a "${LOG_FILE}") 2>&1
  echo "Logging stdout/stderr to ${LOG_FILE}"
fi

echo "====================================================="
echo " Rebuttal 1LAu — Aspirin EVAL ONLY (Multi-Seed)"
echo " SEEDS=${SEEDS} DEVICE=${DEVICE}"
echo " CKPT_DIR=${CKPT_DIR}"
echo " OUT_DIR=${OUT_DIR}"
echo " LOG_FILE=${LOG_FILE}"
echo "====================================================="

for run_tag in "${TAGS[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    koop_ckpt="${CKPT_DIR}/graph_aware_koopman_aspirin_seed${seed}_${run_tag}_best.pt"
    gru_ckpt="${CKPT_DIR}/graph_aware_gru_aspirin_seed${seed}_${run_tag}_best.pt"
    flat_ckpt="${CKPT_DIR}/flat_koopman_aspirin_seed${seed}_${run_tag}_best.pt"

    if [[ ! -f "${koop_ckpt}" || ! -f "${gru_ckpt}" ]]; then
      echo "SKIP missing koop/gru ckpt for tag=${run_tag} seed=${seed}"
      continue
    fi

    flat_args=()
    if [[ -f "${flat_ckpt}" ]]; then
      flat_args+=(--flat-ckpt "${flat_ckpt}")
      echo " Using flat ckpt for Phase-9 R_edge: ${flat_ckpt}"
    else
      echo " WARN: no flat ckpt for tag=${run_tag} seed=${seed} — R_edge Option B will be missing"
    fi

    echo "------------------------------------------------------------"
    echo " EVAL run_tag=${run_tag} seed=${seed}"
    echo "------------------------------------------------------------"

    "${CLI[@]}" eval \
      --md17 aspirin \
      --koopman-ckpt "${koop_ckpt}" \
      --gru-ckpt "${gru_ckpt}" \
      "${flat_args[@]}" \
      --device "${DEVICE}" \
      --rollout-steps "${ROLLOUT_STEPS}" \
      --out-dir "${OUT_DIR}/${run_tag}/seed${seed}"
  done
done

echo "====================================================="
echo " Eval-only complete. Log should include for each seed/tag:"
echo "   LATENT ENERGY RATIO"
echo "   PHYSICAL COORDINATE EDGE LENGTH RATIO (OPTION B)"
echo " Log file: ${LOG_FILE}"
ls -la "${LOG_FILE}" || true
echo "====================================================="
