#!/bin/bash
# ==============================================================================
# Q1 — G-GRU stabilizers: pushforward + training noise (no symplectic)
#
# Paper G-GRU already uses unroll_steps=4 and noise=0. This script trains
# G-GRU under the SAME full loss as KGE, with longer pushforward and/or
# latent training noise, then evals bond/angle/torsion/R_norm/R_edge vs
# existing KGE full checkpoints.
#
# Modes:
#   MODE=combined  (default) — UNROLL=16 + TRAIN_NOISE_STD=0.01 together
#   MODE=matrix              — three arms for attribution:
#                                (1) pf-only   UNROLL=16 noise=0
#                                (2) noise-only UNROLL=4  noise=0.01
#                                (3) combined  UNROLL=16 noise=0.01
#   MODE=custom              — use UNROLL / TRAIN_NOISE_STD env as-is
#
# Usage (GCP, repo root; tmux recommended):
#   DEVICE=cuda SEEDS=42 ./scripts/run_q1_pushforward.sh
#   DEVICE=cuda SEEDS=42 MODE=matrix ./scripts/run_q1_pushforward.sh
#   DEVICE=cuda SEEDS="42 1337 2026" ./scripts/run_q1_pushforward.sh
# ==============================================================================

set -euo pipefail

EPOCHS="${EPOCHS:-100}"
SEEDS="${SEEDS:-42}"
MODE="${MODE:-combined}"
UNROLL="${UNROLL:-16}"
TRAIN_NOISE_STD="${TRAIN_NOISE_STD:-0.01}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MOL="${MOL:-aspirin}"
CKPT_DIR="${CKPT_DIR:-./checkpoints/rebuttal_1LAu}"
OUT_DIR="${OUT_DIR:-./results/rebuttal_q1}"
LOG_FILE="${LOG_FILE:-eval_logs/rebuttal_q1_stabilizers.log}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
KGE_TAG="${KGE_TAG:-full}"
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

mkdir -p "${CKPT_DIR}" "${OUT_DIR}" "$(dirname "${LOG_FILE}")"
LOG_FILE="$(cd "$(dirname "${LOG_FILE}")" && pwd)/$(basename "${LOG_FILE}")"

if [[ -z "${_Q1_LOGGING:-}" ]]; then
  export _Q1_LOGGING=1
  : > "${LOG_FILE}"
  exec > >(tee -a "${LOG_FILE}") 2>&1
  echo "Logging to ${LOG_FILE}"
fi

make_tag() {
  local unroll="$1"
  local noise="$2"
  if [[ "${noise}" == "0" || "${noise}" == "0.0" || "${noise}" == "0.00" ]]; then
    echo "pf${unroll}"
  else
    local noise_tag
    noise_tag=$(printf '%s' "${noise}" | tr -d '.')
    echo "pf${unroll}n${noise_tag}"
  fi
}

# Build list of (unroll,noise) pairs for this MODE
ARMS=()
case "${MODE}" in
  combined)
    ARMS+=("${UNROLL}:${TRAIN_NOISE_STD}")
    ;;
  matrix)
    # Attribution: pushforward alone, noise alone, both
    ARMS+=("16:0.0")
    ARMS+=("4:0.01")
    ARMS+=("16:0.01")
    ;;
  custom)
    ARMS+=("${UNROLL}:${TRAIN_NOISE_STD}")
    ;;
  *)
    echo "Unknown MODE=${MODE} (use combined|matrix|custom)"
    exit 1
    ;;
esac

train_eval_arm() {
  local unroll="$1"
  local noise="$2"
  local seed="$3"
  local run_tag
  run_tag="$(make_tag "${unroll}" "${noise}")"

  local gru_ckpt="${CKPT_DIR}/graph_aware_gru_${MOL}_seed${seed}_${run_tag}_best.pt"
  local koop_ckpt="${CKPT_DIR}/graph_aware_koopman_${MOL}_seed${seed}_${KGE_TAG}_best.pt"
  local base_gru="${CKPT_DIR}/graph_aware_gru_${MOL}_seed${seed}_${BASE_GRU_TAG}_best.pt"

  if [[ ! -f "${koop_ckpt}" ]]; then
    echo "ERROR: missing KGE checkpoint ${koop_ckpt}"
    exit 1
  fi

  if [[ "${SKIP_EXISTING}" == "1" && -f "${gru_ckpt}" ]]; then
    echo "SKIP train (exists): ${gru_ckpt}"
  else
    echo "------------------------------------------------------------"
    echo " TRAIN G-GRU ${MOL} seed=${seed} tag=${run_tag} unroll=${unroll} noise=${noise}"
    echo "------------------------------------------------------------"
    "${CLI[@]}" train \
      --md17 "${MOL}" \
      --model gru \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --seed "${seed}" \
      --device "${DEVICE}" \
      --out-dir "${CKPT_DIR}" \
      --run-tag "${run_tag}" \
      --unroll-steps "${unroll}" \
      --train-noise-std "${noise}" \
      --lambda-dyn 1.0 \
      --lambda-recon 10.0 \
      --lambda-collapse 2.0 \
      --lambda-iso 5.0
  fi

  echo "------------------------------------------------------------"
  echo " EVAL KGE(${KGE_TAG}) vs G-GRU(${run_tag}) seed=${seed}"
  echo "------------------------------------------------------------"
  "${CLI[@]}" eval \
    --md17 "${MOL}" \
    --koopman-ckpt "${koop_ckpt}" \
    --gru-ckpt "${gru_ckpt}" \
    --device "${DEVICE}" \
    --rollout-steps 29 \
    --out-dir "${OUT_DIR}/${MOL}_${run_tag}/seed${seed}"

  if [[ "${INCLUDE_BASE_GRU}" == "1" && -f "${base_gru}" && "${BASE_GRU_TAG}" != "${run_tag}" ]]; then
    # Reference eval once per seed (paper GRU), not once per arm
    local ref_dir="${OUT_DIR}/${MOL}_${BASE_GRU_TAG}_ref/seed${seed}"
    if [[ ! -d "${ref_dir}" ]]; then
      echo "------------------------------------------------------------"
      echo " EVAL reference: KGE(${KGE_TAG}) vs paper G-GRU(${BASE_GRU_TAG}) seed=${seed}"
      echo "------------------------------------------------------------"
      "${CLI[@]}" eval \
        --md17 "${MOL}" \
        --koopman-ckpt "${koop_ckpt}" \
        --gru-ckpt "${base_gru}" \
        --device "${DEVICE}" \
        --rollout-steps 29 \
        --out-dir "${ref_dir}"
    fi
  fi
}

echo "====================================================="
echo " Q1 G-GRU stabilizers — pushforward + training noise"
echo " MODE=${MODE} MOL=${MOL} SEEDS=${SEEDS} EPOCHS=${EPOCHS}"
echo " DEVICE=${DEVICE} (no symplectic)"
echo " Arms:"
for arm in "${ARMS[@]}"; do
  u="${arm%%:*}"
  n="${arm##*:}"
  echo "   - $(make_tag "${u}" "${n}")  (unroll=${u}, noise=${n})"
done
echo " vs KGE tag=${KGE_TAG} (existing; not retrained)"
echo "====================================================="

for seed in "${SEED_LIST[@]}"; do
  for arm in "${ARMS[@]}"; do
    u="${arm%%:*}"
    n="${arm##*:}"
    train_eval_arm "${u}" "${n}" "${seed}"
  done
done

echo "====================================================="
echo " Q1 stabilizer sweep done."
echo " Results under ${OUT_DIR}/${MOL}_*/seed*/"
echo " Log: ${LOG_FILE}"
echo ""
echo " Decision guide (vs KGE full on bond/angle/torsion/R_norm):"
echo "   A) combined (pf+noise) closes most of the gap -> SO(n) not uniquely necessary"
echo "   B) gap shrinks but KGE still wins -> SO(n) still adds something under this protocol"
echo "   C) little change -> try UNROLL=29 or MODE=matrix if you only ran combined"
echo "====================================================="
