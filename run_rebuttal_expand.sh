#!/bin/bash
# ==============================================================================
# Follow-up multi-seed sweep AFTER aspirin pilot validates Outcome A or B.
# Do NOT run until aspirin noreg results are classified.
#
# Seeds default to paper protocol {42, 1337, 2026}.
#
# Usage (log is written automatically; do NOT wrap in an outer tee):
#   OUTCOME=B ./run_rebuttal_expand.sh
#
# Env knobs:
#   OUTCOME=A|B SEEDS="42 1337 2026" DEVICE=cuda INCLUDE_FLAT=1 EPOCHS=100
#   LOG_FILE=eval_logs/rebuttal_1LAu_expand.log
# ==============================================================================

set -euo pipefail

OUTCOME="${OUTCOME:-}"
if [[ -z "${OUTCOME}" ]]; then
  echo "Set OUTCOME=A or OUTCOME=B based on the aspirin pilot before expanding."
  exit 1
fi

EPOCHS="${EPOCHS:-100}"
SEEDS="${SEEDS:-42 1337 2026}"
DEVICE="${DEVICE:-cuda}"
INCLUDE_FLAT="${INCLUDE_FLAT:-1}"
BATCH_MD17="${BATCH_MD17:-32}"
BATCH_MD22="${BATCH_MD22:-16}"
CKPT_DIR="${CKPT_DIR:-./checkpoints/rebuttal_1LAu}"
OUT_DIR="${OUT_DIR:-./results/rebuttal_1LAu}"
LOG_FILE="${LOG_FILE:-eval_logs/rebuttal_1LAu_expand.log}"
PYTHON="${PYTHON:-python3}"
export PYTHONUNBUFFERED=1
CLI=(${PYTHON} -u -m koopman_evolver.cli)

# shellcheck disable=SC2206
SEED_LIST=(${SEEDS})

mkdir -p "${CKPT_DIR}" "${OUT_DIR}" "$(dirname "${LOG_FILE}")"

# Built-in tee so the log is always on disk (last aspirin run missed this).
LOG_FILE="$(cd "$(dirname "${LOG_FILE}")" && pwd)/$(basename "${LOG_FILE}")"
if [[ -z "${_REBUTTAL_LOGGING:-}" ]]; then
  export _REBUTTAL_LOGGING=1
  : > "${LOG_FILE}"
  exec > >(tee -a "${LOG_FILE}") 2>&1
  echo "Logging stdout/stderr to ${LOG_FILE}"
fi

# Conservative expansion set for A/B
MD17_MOLS=(malonaldehyde benzene ethanol)
MD22_MOLS=(dha at-at)

if [[ "${OUTCOME}" == "A" ]]; then
  # Strong architectural signal: cover more systems
  MD17_MOLS=(malonaldehyde benzene ethanol naphthalene salicylic toluene uracil)
  MD22_MOLS=(stachyose ac-ala3-nhme dha at-at)
fi

train_eval_mol_seed() {
  local dataset_flag="$1"   # --md17 or --md22
  local mol="$2"
  local batch="$3"
  local run_tag="$4"
  local seed="$5"
  local lam_c="$6"
  local lam_i="$7"

  local models=(koopman gru)
  if [[ "${INCLUDE_FLAT}" == "1" ]]; then
    models+=(flat)
  fi

  for model in "${models[@]}"; do
    echo "TRAIN ${dataset_flag} ${mol} model=${model} tag=${run_tag} seed=${seed}"
    "${CLI[@]}" train \
      ${dataset_flag} "${mol}" \
      --model "${model}" \
      --epochs "${EPOCHS}" \
      --batch-size "${batch}" \
      --seed "${seed}" \
      --device "${DEVICE}" \
      --out-dir "${CKPT_DIR}" \
      --run-tag "${run_tag}" \
      --lambda-dyn 1.0 \
      --lambda-recon 10.0 \
      --lambda-collapse "${lam_c}" \
      --lambda-iso "${lam_i}"
  done

  local prefix=""
  if [[ "${dataset_flag}" == "--md17" ]]; then
    prefix="md17"
  else
    prefix="md22"
  fi

  local flat_args=()
  if [[ "${INCLUDE_FLAT}" == "1" ]]; then
    flat_args+=(--flat-ckpt "${CKPT_DIR}/flat_koopman_${mol}_seed${seed}_${run_tag}_best.pt")
  fi

  echo "EVAL ${dataset_flag} ${mol} tag=${run_tag} seed=${seed}"
  "${CLI[@]}" eval \
    ${dataset_flag} "${mol}" \
    --koopman-ckpt "${CKPT_DIR}/graph_aware_koopman_${mol}_seed${seed}_${run_tag}_best.pt" \
    --gru-ckpt "${CKPT_DIR}/graph_aware_gru_${mol}_seed${seed}_${run_tag}_best.pt" \
    "${flat_args[@]}" \
    --device "${DEVICE}" \
    --rollout-steps 29 \
    --out-dir "${OUT_DIR}/${prefix}_${mol}/${run_tag}/seed${seed}"
}

echo "====================================================="
echo " Rebuttal expansion (OUTCOME=${OUTCOME}, SEEDS=${SEEDS}, DEVICE=${DEVICE}, INCLUDE_FLAT=${INCLUDE_FLAT})"
echo "====================================================="

for mol in "${MD17_MOLS[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    train_eval_mol_seed --md17 "${mol}" "${BATCH_MD17}" noreg "${seed}" 0.0 0.0
  done
done

for mol in "${MD22_MOLS[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    train_eval_mol_seed --md22 "${mol}" "${BATCH_MD22}" noreg "${seed}" 0.0 0.0
  done
done

echo "====================================================="
echo " Expansion complete (multi-seed)."
echo " Results under ${OUT_DIR}/<system>/noreg/seed{42,1337,2026}/"
echo " Log file: ${LOG_FILE}"
ls -la "${LOG_FILE}" || true
echo " Aggregate mean±std across seeds from that log."
echo "====================================================="
