#!/bin/bash
# ==============================================================================
# Q1 — G-GRU stabilizers: pushforward + training noise (no symplectic, no N-body)
#
# Trains G-GRU only under the SAME full loss as KGE. KGE full ckpts are reused
# (not retrained). Covers paper MD17 + MD22 by default.
#
# Modes:
#   MODE=matrix    (default) — three arms:
#                                (1) pf-only   UNROLL=16 noise=0
#                                (2) noise-only UNROLL=4  noise=0.01
#                                (3) combined  UNROLL=16 noise=0.01
#   MODE=combined            — single arm from UNROLL + TRAIN_NOISE_STD
#   MODE=custom              — same as combined
#
# Protocol (recommended):
#   1) SEEDS=42 MODE=matrix across all MD17/MD22  (~36 GRU trains)
#   2) Expand SEEDS="42 1337 2026" only on winning arm(s)  (MODE=combined/custom)
#
# Usage:
#   DEVICE=cuda SEEDS=42 ./scripts/run_q1_pushforward.sh
#   DEVICE=cuda SEEDS="42 1337 2026" MODE=combined UNROLL=16 TRAIN_NOISE_STD=0.01 ./scripts/run_q1_pushforward.sh
#   MOL=aspirin DEVICE=cuda SEEDS=42 ./scripts/run_q1_pushforward.sh   # single-mol override
# ==============================================================================

set -euo pipefail

EPOCHS="${EPOCHS:-100}"
SEEDS="${SEEDS:-42}"
MODE="${MODE:-matrix}"
UNROLL="${UNROLL:-16}"
TRAIN_NOISE_STD="${TRAIN_NOISE_STD:-0.01}"
DEVICE="${DEVICE:-cuda}"
BATCH_MD17="${BATCH_MD17:-32}"
BATCH_MD22="${BATCH_MD22:-16}"
# MOL="" or unset => all paper MD17/MD22; else space-separated list
MOL="${MOL:-}"
CKPT_DIR="${CKPT_DIR:-./checkpoints/rebuttal_1LAu}"
OUT_DIR="${OUT_DIR:-./results/rebuttal_q1}"
LOG_FILE="${LOG_FILE:-eval_logs/rebuttal_q1_stabilizers.log}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SKIP_MISSING_KGE="${SKIP_MISSING_KGE:-1}"
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

MD17_ALL=(aspirin benzene ethanol malonaldehyde naphthalene salicylic toluene uracil)
MD22_ALL=(stachyose ac-ala3-nhme dha at-at)

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

is_md22() {
  local mol="$1"
  case "${mol}" in
    stachyose|ac-ala3-nhme|dha|at-at|at-at-cg|buckyball-catcher|dw-nanotube) return 0 ;;
    *) return 1 ;;
  esac
}

# Resolve molecule list: (dataset_flag, mol, batch)
JOBS=()
if [[ -n "${MOL}" ]]; then
  # shellcheck disable=SC2206
  MOL_LIST=(${MOL})
  for mol in "${MOL_LIST[@]}"; do
    if is_md22 "${mol}"; then
      JOBS+=("--md22:${mol}:${BATCH_MD22}")
    else
      JOBS+=("--md17:${mol}:${BATCH_MD17}")
    fi
  done
else
  for mol in "${MD17_ALL[@]}"; do
    JOBS+=("--md17:${mol}:${BATCH_MD17}")
  done
  for mol in "${MD22_ALL[@]}"; do
    JOBS+=("--md22:${mol}:${BATCH_MD22}")
  done
fi

ARMS=()
case "${MODE}" in
  combined|custom)
    ARMS+=("${UNROLL}:${TRAIN_NOISE_STD}")
    ;;
  matrix)
    ARMS+=("16:0.0")
    ARMS+=("4:0.01")
    ARMS+=("16:0.01")
    ;;
  *)
    echo "Unknown MODE=${MODE} (use combined|matrix|custom)"
    exit 1
    ;;
esac

train_eval_arm() {
  local dataset_flag="$1"
  local mol="$2"
  local batch="$3"
  local unroll="$4"
  local noise="$5"
  local seed="$6"
  local run_tag
  run_tag="$(make_tag "${unroll}" "${noise}")"

  local gru_ckpt="${CKPT_DIR}/graph_aware_gru_${mol}_seed${seed}_${run_tag}_best.pt"
  local koop_ckpt="${CKPT_DIR}/graph_aware_koopman_${mol}_seed${seed}_${KGE_TAG}_best.pt"
  local base_gru="${CKPT_DIR}/graph_aware_gru_${mol}_seed${seed}_${BASE_GRU_TAG}_best.pt"

  if [[ ! -f "${koop_ckpt}" ]]; then
    if [[ "${SKIP_MISSING_KGE}" == "1" ]]; then
      echo "SKIP missing KGE: ${koop_ckpt}"
      return 0
    fi
    echo "ERROR: missing KGE checkpoint ${koop_ckpt}"
    exit 1
  fi

  if [[ "${SKIP_EXISTING}" == "1" && -f "${gru_ckpt}" ]]; then
    echo "SKIP train (exists): ${gru_ckpt}"
  else
    echo "------------------------------------------------------------"
    echo " TRAIN G-GRU ${dataset_flag} ${mol} seed=${seed} tag=${run_tag} unroll=${unroll} noise=${noise}"
    echo "------------------------------------------------------------"
    "${CLI[@]}" train \
      ${dataset_flag} "${mol}" \
      --model gru \
      --epochs "${EPOCHS}" \
      --batch-size "${batch}" \
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
  echo " EVAL KGE(${KGE_TAG}) vs G-GRU(${run_tag}) ${mol} seed=${seed}"
  echo "------------------------------------------------------------"
  local flat_args=()
  local flat_ckpt="${CKPT_DIR}/flat_koopman_${mol}_seed${seed}_${KGE_TAG}_best.pt"
  if [[ -f "${flat_ckpt}" ]]; then
    flat_args+=(--flat-ckpt "${flat_ckpt}")
    echo "   (+ Phase-9 R_norm/R_edge via flat ${flat_ckpt})"
  fi
  "${CLI[@]}" eval \
    ${dataset_flag} "${mol}" \
    --koopman-ckpt "${koop_ckpt}" \
    --gru-ckpt "${gru_ckpt}" \
    "${flat_args[@]}" \
    --device "${DEVICE}" \
    --rollout-steps 29 \
    --out-dir "${OUT_DIR}/${mol}_${run_tag}/seed${seed}"

  if [[ "${INCLUDE_BASE_GRU}" == "1" && -f "${base_gru}" && "${BASE_GRU_TAG}" != "${run_tag}" ]]; then
    local ref_dir="${OUT_DIR}/${mol}_${BASE_GRU_TAG}_ref/seed${seed}"
    if [[ ! -d "${ref_dir}" ]]; then
      echo "------------------------------------------------------------"
      echo " EVAL reference: KGE(${KGE_TAG}) vs paper G-GRU(${BASE_GRU_TAG}) ${mol} seed=${seed}"
      echo "------------------------------------------------------------"
      local flat_args=()
      local flat_ckpt="${CKPT_DIR}/flat_koopman_${mol}_seed${seed}_${KGE_TAG}_best.pt"
      if [[ -f "${flat_ckpt}" ]]; then
        flat_args+=(--flat-ckpt "${flat_ckpt}")
      fi
      "${CLI[@]}" eval \
        ${dataset_flag} "${mol}" \
        --koopman-ckpt "${koop_ckpt}" \
        --gru-ckpt "${base_gru}" \
        "${flat_args[@]}" \
        --device "${DEVICE}" \
        --rollout-steps 29 \
        --out-dir "${ref_dir}"
    fi
  fi
}

echo "====================================================="
echo " Q1 G-GRU stabilizers — pushforward + training noise"
echo " MODE=${MODE} SEEDS=${SEEDS} EPOCHS=${EPOCHS} DEVICE=${DEVICE}"
echo " Molecules (${#JOBS[@]}):"
for job in "${JOBS[@]}"; do
  echo "   ${job}"
done
echo " Arms:"
for arm in "${ARMS[@]}"; do
  u="${arm%%:*}"
  n="${arm##*:}"
  echo "   - $(make_tag "${u}" "${n}")  (unroll=${u}, noise=${n})"
done
echo " Train G-GRU only; KGE tag=${KGE_TAG} reused. No N-body / no symplectic."
echo " Approx jobs: $((${#JOBS[@]} * ${#SEED_LIST[@]} * ${#ARMS[@]})) GRU trains"
echo "====================================================="

for job in "${JOBS[@]}"; do
  dataset_flag="${job%%:*}"
  rest="${job#*:}"
  mol="${rest%%:*}"
  batch="${rest##*:}"
  for seed in "${SEED_LIST[@]}"; do
    for arm in "${ARMS[@]}"; do
      u="${arm%%:*}"
      n="${arm##*:}"
      train_eval_arm "${dataset_flag}" "${mol}" "${batch}" "${u}" "${n}" "${seed}"
    done
  done
done

echo "====================================================="
echo " Q1 stabilizer sweep done."
echo " Results under ${OUT_DIR}/<mol>_<tag>/seed*/"
echo " Log: ${LOG_FILE}"
echo ""
echo " Next: download log, pick winning arm(s), then expand seeds, e.g."
echo "   DEVICE=cuda SEEDS=\"42 1337 2026\" MODE=combined UNROLL=16 TRAIN_NOISE_STD=0.01 ./scripts/run_q1_pushforward.sh"
echo "====================================================="
