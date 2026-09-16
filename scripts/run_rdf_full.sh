#!/bin/bash
# ==============================================================================
# RDF pilot over all *full* KGE/G-GRU checkpoints under CKPT_DIR.
# Skips noreg. Discovers molecules/seeds from koopman checkpoint names.
#
# Usage (GCP, repo root):
#   ./scripts/run_rdf_full.sh
#   DEVICE=cuda CKPT_DIR=./checkpoints/rebuttal_1LAu ./scripts/run_rdf_full.sh
# ==============================================================================

set -euo pipefail

CKPT_DIR="${CKPT_DIR:-./checkpoints/rebuttal_1LAu}"
OUT_ROOT="${OUT_ROOT:-./results/rdf_pilot}"
DEVICE="${DEVICE:-cuda}"
PYTHON="${PYTHON:-python3}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-29}"
LOG_FILE="${LOG_FILE:-eval_logs/rdf_pilot_full.log}"

# Prefer repo-root imports (same as `python -m koopman_evolver.cli`)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUT_ROOT}" "$(dirname "${LOG_FILE}")"
LOG_FILE="$(cd "$(dirname "${LOG_FILE}")" && pwd)/$(basename "${LOG_FILE}")"

if [[ -z "${_RDF_LOGGING:-}" ]]; then
  export _RDF_LOGGING=1
  export PYTHONUNBUFFERED=1
  : > "${LOG_FILE}"
  exec > >(tee -a "${LOG_FILE}") 2>&1
  echo "Logging to ${LOG_FILE}"
fi

echo "====================================================="
echo " RDF pilot — FULL checkpoints only"
echo " CKPT_DIR=${CKPT_DIR} DEVICE=${DEVICE} OUT_ROOT=${OUT_ROOT}"
echo "====================================================="

shopt -s nullglob
koop_ckpts=("${CKPT_DIR}"/graph_aware_koopman_*_full_best.pt)
if [[ ${#koop_ckpts[@]} -eq 0 ]]; then
  echo "No graph_aware_koopman_*_full_best.pt under ${CKPT_DIR}"
  exit 1
fi

# Sort for stable order
IFS=$'\n' koop_ckpts=($(printf '%s\n' "${koop_ckpts[@]}" | sort)); unset IFS

fail=0
ok=0
for koop in "${koop_ckpts[@]}"; do
  base="$(basename "${koop}")"
  # graph_aware_koopman_<mol>_seed<seed>_full_best.pt
  if [[ ! "${base}" =~ ^graph_aware_koopman_(.+)_seed([0-9]+)_full_best\.pt$ ]]; then
    echo "SKIP unrecognized name: ${base}"
    continue
  fi
  mol="${BASH_REMATCH[1]}"
  seed="${BASH_REMATCH[2]}"
  gru="${CKPT_DIR}/graph_aware_gru_${mol}_seed${seed}_full_best.pt"
  if [[ ! -f "${gru}" ]]; then
    echo "SKIP missing GRU: ${gru}"
    ((fail++)) || true
    continue
  fi

  # Heuristic: md22 molecules used in expand + paper MD22 set
  dataset_flag="--md17"
  case "${mol}" in
    stachyose|ac-ala3-nhme|dha|at-at|at-at-cg|buckyball-catcher|dw-nanotube)
      dataset_flag="--md22"
      ;;
  esac

  out_dir="${OUT_ROOT}/${mol}_full_seed${seed}"
  echo "------------------------------------------------------------"
  echo " RDF ${dataset_flag} ${mol} seed=${seed}"
  echo "   KGE: ${koop}"
  echo "   GRU: ${gru}"
  echo "   OUT: ${out_dir}"
  echo "------------------------------------------------------------"

  if "${PYTHON}" -u scripts/rdf_from_ckpt.py \
      ${dataset_flag} "${mol}" \
      --koopman-ckpt "${koop}" \
      --gru-ckpt "${gru}" \
      --device "${DEVICE}" \
      --rollout-steps "${ROLLOUT_STEPS}" \
      --out-dir "${out_dir}"; then
    ((ok++)) || true
  else
    echo "FAIL ${mol} seed=${seed}"
    ((fail++)) || true
  fi
done

echo "====================================================="
echo " RDF full sweep done. ok=${ok} fail=${fail}"
echo " Log: ${LOG_FILE}"
echo " Metrics under ${OUT_ROOT}/*/rdf_*_metrics.json"
echo "====================================================="
[[ "${fail}" -eq 0 ]]
