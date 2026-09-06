#!/bin/bash
# ==============================================================================
# Train FULL-loss KGE + G-GRU on MD17/MD22 for RDF sweep (no N-body, no flat).
#
# Paper loss: lambda_collapse=2, lambda_iso=5, lambda_recon=10, lambda_dyn=1
# Skips checkpoints that already exist (aspirin full × 3 seeds already on VM).
# Does NOT run PhysicsEval — use scripts/run_rdf_full.sh after training.
#
# Usage (GCP, repo root):
#   chmod +x scripts/run_rdf_train_full.sh
#   DEVICE=cuda ./scripts/run_rdf_train_full.sh
#
# Env knobs:
#   EPOCHS=100 SEEDS="42 1337 2026" DEVICE=cuda
#   BATCH_MD17=32 BATCH_MD22=16 SKIP_EXISTING=1
#   SCOPE=paper|expand   # paper = 8 MD17 + 4 MD22; expand = Outcome-A set
#   CKPT_DIR=./checkpoints/rebuttal_1LAu
#   LOG_FILE=eval_logs/rdf_train_full.log
# ==============================================================================

set -euo pipefail

EPOCHS="${EPOCHS:-100}"
SEEDS="${SEEDS:-42 1337 2026}"
DEVICE="${DEVICE:-cuda}"
BATCH_MD17="${BATCH_MD17:-32}"
BATCH_MD22="${BATCH_MD22:-16}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SCOPE="${SCOPE:-paper}"
CKPT_DIR="${CKPT_DIR:-./checkpoints/rebuttal_1LAu}"
LOG_FILE="${LOG_FILE:-eval_logs/rdf_train_full.log}"
RUN_TAG="${RUN_TAG:-full}"
LAM_COLLAPSE="${LAM_COLLAPSE:-2.0}"
LAM_ISO="${LAM_ISO:-5.0}"
PYTHON="${PYTHON:-python3}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

# shellcheck disable=SC2206
SEED_LIST=(${SEEDS})
CLI=(${PYTHON} -u -m koopman_evolver.cli)

mkdir -p "${CKPT_DIR}" "$(dirname "${LOG_FILE}")"
LOG_FILE="$(cd "$(dirname "${LOG_FILE}")" && pwd)/$(basename "${LOG_FILE}")"

if [[ -z "${_RDF_TRAIN_LOGGING:-}" ]]; then
  export _RDF_TRAIN_LOGGING=1
  : > "${LOG_FILE}"
  exec > >(tee -a "${LOG_FILE}") 2>&1
  echo "Logging to ${LOG_FILE}"
fi

# Paper Table / run_all.sh molecule set (no N-body)
MD17_MOLS=(aspirin benzene ethanol malonaldehyde naphthalene salicylic toluene uracil)
MD22_MOLS=(stachyose ac-ala3-nhme dha at-at)

if [[ "${SCOPE}" == "expand" ]]; then
  # Same as Outcome-A expand (still no N-body)
  MD17_MOLS=(aspirin malonaldehyde benzene ethanol naphthalene salicylic toluene uracil)
  MD22_MOLS=(stachyose ac-ala3-nhme dha at-at)
fi

ckpt_path() {
  local model="$1" mol="$2" seed="$3"
  echo "${CKPT_DIR}/graph_aware_${model}_${mol}_seed${seed}_${RUN_TAG}_best.pt"
}

# Returns: 0=trained, 2=skipped, 1=fail
train_one() {
  local dataset_flag="$1"
  local mol="$2"
  local batch="$3"
  local model="$4"
  local seed="$5"
  local ckpt
  ckpt="$(ckpt_path "${model}" "${mol}" "${seed}")"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${ckpt}" ]]; then
    echo "SKIP existing ${ckpt}"
    return 2
  fi

  echo "------------------------------------------------------------"
  echo " TRAIN ${dataset_flag} ${mol} model=${model} seed=${seed} tag=${RUN_TAG}"
  echo "------------------------------------------------------------"
  if "${CLI[@]}" train \
      ${dataset_flag} "${mol}" \
      --model "${model}" \
      --epochs "${EPOCHS}" \
      --batch-size "${batch}" \
      --seed "${seed}" \
      --device "${DEVICE}" \
      --out-dir "${CKPT_DIR}" \
      --run-tag "${RUN_TAG}" \
      --lambda-dyn 1.0 \
      --lambda-recon 10.0 \
      --lambda-collapse "${LAM_COLLAPSE}" \
      --lambda-iso "${LAM_ISO}"; then
    return 0
  fi
  echo "FAIL train ${dataset_flag} ${mol} ${model} seed=${seed}"
  return 1
}

echo "====================================================="
echo " RDF full-loss train — MD17/MD22 only"
echo " SCOPE=${SCOPE} EPOCHS=${EPOCHS} SEEDS=${SEEDS} DEVICE=${DEVICE}"
echo " SKIP_EXISTING=${SKIP_EXISTING} CKPT_DIR=${CKPT_DIR}"
echo " loss: collapse=${LAM_COLLAPSE} iso=${LAM_ISO} (full paper)"
echo " MD17: ${MD17_MOLS[*]}"
echo " MD22: ${MD22_MOLS[*]}"
echo "====================================================="

ok=0
skip=0
fail=0

for mol in "${MD17_MOLS[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    for model in koopman gru; do
      rc=0
      train_one --md17 "${mol}" "${BATCH_MD17}" "${model}" "${seed}" || rc=$?
      if [[ "${rc}" -eq 0 ]]; then ((ok++)) || true
      elif [[ "${rc}" -eq 2 ]]; then ((skip++)) || true
      else ((fail++)) || true
      fi
    done
  done
done

for mol in "${MD22_MOLS[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    for model in koopman gru; do
      rc=0
      train_one --md22 "${mol}" "${BATCH_MD22}" "${model}" "${seed}" || rc=$?
      if [[ "${rc}" -eq 0 ]]; then ((ok++)) || true
      elif [[ "${rc}" -eq 2 ]]; then ((skip++)) || true
      else ((fail++)) || true
      fi
    done
  done
done

echo "====================================================="
echo " Train sweep done. trained_ok=${ok} skipped=${skip} fail=${fail}"
echo " Next: DEVICE=cuda ./scripts/run_rdf_full.sh"
echo " Log: ${LOG_FILE}"
echo "====================================================="
[[ "${fail}" -eq 0 ]]
