#!/bin/bash
# ==============================================================================
# Reviewer 1LAu Rebuttal Ablation — Aspirin Multi-Seed Pilot (GCP)
#
# Goal: Isolate whether long-horizon structural gains of Graph Koopman vs Graph
# GRU come from the orthogonal Koopman transition, or from L_collapse / L_iso.
#
# Design (paired comparison on MD17 aspirin, seeds {42, 1337, 2026}):
#   1) full  : paper loss  (lambda_collapse=2, lambda_iso=5)
#   2) noreg : reviewer ablation (lambda_collapse=0, lambda_iso=0)
#              keeps only L_recon + L_dyn
#
# Models trained per setting × seed: koopman, gru
# Optional: flat (set INCLUDE_FLAT=1)
#
# Usage (on GCP VM, from repo root):
#   chmod +x run_rebuttal_aspirin.sh
#   ./run_rebuttal_aspirin.sh 2>&1 | tee eval_logs/rebuttal_1LAu_aspirin.log
#
# Env knobs:
#   EPOCHS=100 SEEDS="42 1337 2026" BATCH_SIZE=32 INCLUDE_FLAT=0 SKIP_FULL=0
# ==============================================================================

set -euo pipefail

EPOCHS="${EPOCHS:-100}"
# Paper multi-seed protocol (space-separated; override via SEEDS="42" for a smoke test)
SEEDS="${SEEDS:-42 1337 2026}"
BATCH_SIZE="${BATCH_SIZE:-32}"
INCLUDE_FLAT="${INCLUDE_FLAT:-0}"
SKIP_FULL="${SKIP_FULL:-0}"
CKPT_DIR="${CKPT_DIR:-./checkpoints/rebuttal_1LAu}"
OUT_DIR="${OUT_DIR:-./results/rebuttal_1LAu/aspirin}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-29}"

# shellcheck disable=SC2206
SEED_LIST=(${SEEDS})

mkdir -p "${CKPT_DIR}" "${OUT_DIR}" eval_logs

PYTHON="${PYTHON:-python3}"
CLI=(${PYTHON} -u -m koopman_evolver.cli)

train_one() {
  local model="$1"
  local run_tag="$2"
  local seed="$3"
  local lam_collapse="$4"
  local lam_iso="$5"

  echo "------------------------------------------------------------"
  echo " TRAIN model=${model} run_tag=${run_tag} seed=${seed} collapse=${lam_collapse} iso=${lam_iso}"
  echo "------------------------------------------------------------"

  "${CLI[@]}" train \
    --md17 aspirin \
    --model "${model}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --seed "${seed}" \
    --out-dir "${CKPT_DIR}" \
    --run-tag "${run_tag}" \
    --lambda-dyn 1.0 \
    --lambda-recon 10.0 \
    --lambda-collapse "${lam_collapse}" \
    --lambda-iso "${lam_iso}"
}

eval_pair() {
  local run_tag="$1"
  local seed="$2"
  local koop_ckpt="${CKPT_DIR}/graph_aware_koopman_aspirin_seed${seed}_${run_tag}_best.pt"
  local gru_ckpt="${CKPT_DIR}/graph_aware_gru_aspirin_seed${seed}_${run_tag}_best.pt"
  local flat_args=()

  if [[ "${INCLUDE_FLAT}" == "1" ]]; then
    flat_args+=(--flat-ckpt "${CKPT_DIR}/flat_koopman_aspirin_seed${seed}_${run_tag}_best.pt")
  fi

  echo "------------------------------------------------------------"
  echo " EVAL run_tag=${run_tag} seed=${seed}"
  echo "   koop: ${koop_ckpt}"
  echo "   gru : ${gru_ckpt}"
  echo "------------------------------------------------------------"

  "${CLI[@]}" eval \
    --md17 aspirin \
    --koopman-ckpt "${koop_ckpt}" \
    --gru-ckpt "${gru_ckpt}" \
    "${flat_args[@]}" \
    --rollout-steps "${ROLLOUT_STEPS}" \
    --out-dir "${OUT_DIR}/${run_tag}/seed${seed}"
}

run_setting() {
  local run_tag="$1"
  local lam_collapse="$2"
  local lam_iso="$3"

  echo ""
  echo "### SETTING: ${run_tag} (lambda_collapse=${lam_collapse}, lambda_iso=${lam_iso})"
  for seed in "${SEED_LIST[@]}"; do
    echo ""
    echo "==== Seed ${seed} ===="
    for model in "${MODELS[@]}"; do
      train_one "${model}" "${run_tag}" "${seed}" "${lam_collapse}" "${lam_iso}"
    done
    eval_pair "${run_tag}" "${seed}"
  done
}

echo "====================================================="
echo " Rebuttal 1LAu — Aspirin Loss Ablation (Multi-Seed)"
echo " EPOCHS=${EPOCHS} SEEDS=${SEEDS} BATCH_SIZE=${BATCH_SIZE}"
echo " INCLUDE_FLAT=${INCLUDE_FLAT} SKIP_FULL=${SKIP_FULL}"
echo " CKPT_DIR=${CKPT_DIR}"
echo " OUT_DIR=${OUT_DIR}"
echo "====================================================="

MODELS=(koopman gru)
if [[ "${INCLUDE_FLAT}" == "1" ]]; then
  MODELS+=(flat)
fi

# --- Setting A: full paper loss (control) ---
if [[ "${SKIP_FULL}" != "1" ]]; then
  run_setting "full" 2.0 5.0
else
  echo "Skipping full-loss training (SKIP_FULL=1)"
fi

# --- Setting B: reviewer ablation (no collapse / iso) ---
run_setting "noreg" 0.0 0.0

echo ""
echo "====================================================="
echo " Aspirin multi-seed rebuttal ablation finished."
echo " Per-seed plots/logs:"
echo "   ${OUT_DIR}/full/seed{42,1337,2026}/"
echo "   ${OUT_DIR}/noreg/seed{42,1337,2026}/"
echo " Checkpoints:"
echo "   ${CKPT_DIR}/graph_aware_{koopman,gru}_aspirin_seed*_{full,noreg}_best.pt"
echo ""
echo " Aggregate mean±std across seeds from the PHASE 9 / physical"
echo " diagnostics blocks in eval_logs/rebuttal_1LAu_aspirin.log"
echo ""
echo " Decision guide (use noreg mean±std):"
echo "   A) KGE still wins bond/angle/torsion/R_norm under noreg"
echo "      -> expand to MD17 + MD22"
echo "   B) Gap shrinks but angle/torsion/R_norm still favor KGE"
echo "      -> expand carefully; emphasize non-loss metrics"
echo "   C) Advantage largely disappears under noreg"
echo "      -> reframe claim before scaling compute"
echo "====================================================="
