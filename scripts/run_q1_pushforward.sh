#!/bin/bash
# ==============================================================================
# Q1 — G-GRU pushforward ablation (aspirin first)
#
# Context: paper G-GRU already uses unroll_steps=4 in L_dyn. This trains a
# longer pushforward G-GRU under the SAME full loss as KGE, then evaluates
# bond/angle/torsion/R_norm/R_edge vs existing KGE full checkpoints.
#
# Protocol:
#   1) Smoke: SEEDS=42 UNROLL=16
#   2) If promising: SEEDS="42 1337 2026" and/or UNROLL=29
#   3) Optional noise: TRAIN_NOISE_STD=0.01
#
# Usage (GCP, repo root; tmux recommended):
#   DEVICE=cuda SEEDS=42 UNROLL=16 ./scripts/run_q1_pushforward.sh
#   DEVICE=cuda SEEDS="42 1337 2026" UNROLL=16 ./scripts/run_q1_pushforward.sh
# ==============================================================================

set -euo pipefail

EPOCHS="${EPOCHS:-100}"
SEEDS="${SEEDS:-42}"
UNROLL="${UNROLL:-16}"
TRAIN_NOISE_STD="${TRAIN_NOISE_STD:-0.0}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MOL="${MOL:-aspirin}"
CKPT_DIR="${CKPT_DIR:-./checkpoints/rebuttal_1LAu}"
OUT_DIR="${OUT_DIR:-./results/rebuttal_q1}"
LOG_FILE="${LOG_FILE:-eval_logs/rebuttal_q1_pushforward.log}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
# Compare against this KGE tag (paper full loss)
KGE_TAG="${KGE_TAG:-full}"
# Paper GRU (unroll=4) tag for side-by-side, optional
BASE_GRU_TAG="${BASE_GRU_TAG:-full}"
INCLUDE_BASE_GRU="${INCLUDE_BASE_GRU:-1}"
PYTHON="${PYTHON:-python3}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

# shellcheck disable=SC2206
SEED_LIST=(${SEEDS})
CLI=(${PYTHON} -u -m koopman_evolver.cli)

# Tag encodes the ablation so we never overwrite paper `full` GRU
if [[ "${TRAIN_NOISE_STD}" == "0" || "${TRAIN_NOISE_STD}" == "0.0" || "${TRAIN_NOISE_STD}" == "0.00" ]]; then
  RUN_TAG="pf${UNROLL}"
else
  # e.g. pf16n01 for noise 0.01
  NOISE_TAG=$(printf '%s' "${TRAIN_NOISE_STD}" | tr -d '.')
  RUN_TAG="pf${UNROLL}n${NOISE_TAG}"
fi

mkdir -p "${CKPT_DIR}" "${OUT_DIR}" "$(dirname "${LOG_FILE}")"
LOG_FILE="$(cd "$(dirname "${LOG_FILE}")" && pwd)/$(basename "${LOG_FILE}")"

if [[ -z "${_Q1_LOGGING:-}" ]]; then
  export _Q1_LOGGING=1
  : > "${LOG_FILE}"
  exec > >(tee -a "${LOG_FILE}") 2>&1
  echo "Logging to ${LOG_FILE}"
fi

echo "====================================================="
echo " Q1 G-GRU pushforward — ${MOL}"
echo " UNROLL=${UNROLL} NOISE=${TRAIN_NOISE_STD} TAG=${RUN_TAG}"
echo " SEEDS=${SEEDS} EPOCHS=${EPOCHS} DEVICE=${DEVICE}"
echo " vs KGE tag=${KGE_TAG} (existing ckpts; not retrained)"
echo "====================================================="

for seed in "${SEED_LIST[@]}"; do
  gru_ckpt="${CKPT_DIR}/graph_aware_gru_${MOL}_seed${seed}_${RUN_TAG}_best.pt"
  koop_ckpt="${CKPT_DIR}/graph_aware_koopman_${MOL}_seed${seed}_${KGE_TAG}_best.pt"
  base_gru="${CKPT_DIR}/graph_aware_gru_${MOL}_seed${seed}_${BASE_GRU_TAG}_best.pt"

  if [[ ! -f "${koop_ckpt}" ]]; then
    echo "ERROR: missing KGE checkpoint ${koop_ckpt}"
    exit 1
  fi

  if [[ "${SKIP_EXISTING}" == "1" && -f "${gru_ckpt}" ]]; then
    echo "SKIP train (exists): ${gru_ckpt}"
  else
    echo "------------------------------------------------------------"
    echo " TRAIN G-GRU ${MOL} seed=${seed} tag=${RUN_TAG} unroll=${UNROLL}"
    echo "------------------------------------------------------------"
    "${CLI[@]}" train \
      --md17 "${MOL}" \
      --model gru \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --seed "${seed}" \
      --device "${DEVICE}" \
      --out-dir "${CKPT_DIR}" \
      --run-tag "${RUN_TAG}" \
      --unroll-steps "${UNROLL}" \
      --train-noise-std "${TRAIN_NOISE_STD}" \
      --lambda-dyn 1.0 \
      --lambda-recon 10.0 \
      --lambda-collapse 2.0 \
      --lambda-iso 5.0
  fi

  echo "------------------------------------------------------------"
  echo " EVAL KGE(${KGE_TAG}) vs G-GRU(${RUN_TAG}) seed=${seed}"
  echo "------------------------------------------------------------"
  "${CLI[@]}" eval \
    --md17 "${MOL}" \
    --koopman-ckpt "${koop_ckpt}" \
    --gru-ckpt "${gru_ckpt}" \
    --device "${DEVICE}" \
    --rollout-steps 29 \
    --out-dir "${OUT_DIR}/${MOL}_${RUN_TAG}/seed${seed}"

  if [[ "${INCLUDE_BASE_GRU}" == "1" && -f "${base_gru}" && "${BASE_GRU_TAG}" != "${RUN_TAG}" ]]; then
    echo "------------------------------------------------------------"
    echo " EVAL reference: KGE(${KGE_TAG}) vs paper G-GRU(${BASE_GRU_TAG}) seed=${seed}"
    echo "------------------------------------------------------------"
    "${CLI[@]}" eval \
      --md17 "${MOL}" \
      --koopman-ckpt "${koop_ckpt}" \
      --gru-ckpt "${base_gru}" \
      --device "${DEVICE}" \
      --rollout-steps 29 \
      --out-dir "${OUT_DIR}/${MOL}_${BASE_GRU_TAG}_ref/seed${seed}"
  fi
done

echo "====================================================="
echo " Q1 pushforward sweep done."
echo " New GRU ckpts: ${CKPT_DIR}/graph_aware_gru_${MOL}_seed*_${RUN_TAG}_best.pt"
echo " Results: ${OUT_DIR}/${MOL}_${RUN_TAG}/seed*/"
echo " Log: ${LOG_FILE}"
echo ""
echo " Decision guide (bond/angle/torsion/R_norm vs KGE full):"
echo "   A) G-GRU+pf closes most of the gap -> SO(n) not uniquely necessary here"
echo "   B) Gap shrinks but KGE still wins structure -> SO(n) still adds something"
echo "   C) Little change vs paper GRU (pf4) -> try UNROLL=29 and/or TRAIN_NOISE_STD=0.01"
echo "====================================================="
