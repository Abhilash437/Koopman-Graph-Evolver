#!/bin/bash
# ==============================================================================
# Koopman Graph Evolver - Full Sweep Automation Script
#
# This script sequentially trains all major model architectures across the 
# supported physics and molecular datasets. It is designed to be run unattended 
# on a cloud VM (e.g., GCP with an NVIDIA T4/L4).
# ==============================================================================

# Stop on errors
set -e

# Default hyperparameters
EPOCHS=100
BATCH_SIZE=32

echo "====================================================="
echo " Starting Koopman Graph Evolver Massive Sweep"
echo "====================================================="

# 1. MD17 - Small Molecules (Ethanol, Aspirin)
for mol in ethanol aspirin; do
    echo "--- Training MD17: $mol ---"
    for model in koopman gru flat e-gkn egnn; do
        echo "Running Model: $model"
        python -u -m koopman_evolver.cli train --md17 $mol --model $model --epochs $EPOCHS --batch-size $BATCH_SIZE
    done
    
    echo "--- Evaluating MD17: $mol ---"
    python -u -m koopman_evolver.cli eval --md17 $mol \
        --koopman-ckpt ./checkpoints/graph_aware_koopman_${mol}_best.pt \
        --gru-ckpt ./checkpoints/graph_aware_gru_${mol}_best.pt \
        --flat-ckpt ./checkpoints/flat_koopman_${mol}_best.pt \
        --egkn-ckpt ./checkpoints/e_gkn_${mol}_best.pt \
        --egnn-ckpt ./checkpoints/egnn_${mol}_best.pt
done

# 2. MD22 - Large Macromolecules (Stachyose)
# We lower the batch size to 16 for Stachyose (87 atoms) to prevent VRAM spikes
MD22_BATCH=16
for mol in stachyose; do
    echo "--- Training MD22: $mol ---"
    for model in koopman gru flat e-gkn egnn; do
        echo "Running Model: $model"
        python -u -m koopman_evolver.cli train --md22 $mol --model $model --epochs $EPOCHS --batch-size $MD22_BATCH
    done
    
    echo "--- Evaluating MD22: $mol ---"
    python -u -m koopman_evolver.cli eval --md22 $mol \
        --koopman-ckpt ./checkpoints/graph_aware_koopman_${mol}_best.pt \
        --gru-ckpt ./checkpoints/graph_aware_gru_${mol}_best.pt \
        --flat-ckpt ./checkpoints/flat_koopman_${mol}_best.pt \
        --egkn-ckpt ./checkpoints/e_gkn_${mol}_best.pt \
        --egnn-ckpt ./checkpoints/egnn_${mol}_best.pt
done

# 3. N-Body Physics (Charged)
echo "--- Training N-Body: Charged ---"
for model in e-gkn egnn koopman gru flat; do
    echo "Running Model: $model"
    python -u -m koopman_evolver.cli train --nbody charged --model $model --epochs $EPOCHS --batch-size $BATCH_SIZE
done

echo "--- Evaluating N-Body: Charged ---"
python -u -m koopman_evolver.cli eval --nbody charged \
    --koopman-ckpt ./checkpoints/graph_aware_koopman_charged_best.pt \
    --gru-ckpt ./checkpoints/graph_aware_gru_charged_best.pt \
    --flat-ckpt ./checkpoints/flat_koopman_charged_best.pt \
    --egkn-ckpt ./checkpoints/e_gkn_charged_best.pt \
    --egnn-ckpt ./checkpoints/egnn_charged_best.pt

# 4. METR-LA Macro-Traffic 
# We reduce the epochs to 50 for Traffic as 100 epochs on a 207-node graph takes a very long time
TRAFFIC_EPOCHS=50
echo "--- Training Traffic: METR-LA ---"
for model in gru koopman flat; do
    echo "Running Model: $model"
    python -u -m koopman_evolver.cli train --traffic --model $model --epochs $TRAFFIC_EPOCHS --batch-size $BATCH_SIZE
done

echo "--- Evaluating Traffic: METR-LA ---"
python -u -m koopman_evolver.cli eval --traffic \
    --koopman-ckpt ./checkpoints/graph_aware_koopman_metr-la_best.pt \
    --gru-ckpt ./checkpoints/graph_aware_gru_metr-la_best.pt \
    --flat-ckpt ./checkpoints/flat_koopman_metr-la_best.pt

echo "====================================================="
echo " ALL TRAINING AND EVALUATIONS COMPLETED SUCCESSFULLY!"
echo " Checkpoints are available in ./checkpoints/"
echo "====================================================="
