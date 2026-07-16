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

# 1. MD17 - Small Molecules
for mol in aspirin benzene ethanol malonaldehyde naphthalene salicylic toluene uracil; do
    echo "--- Training MD17: $mol ---"
    for model in koopman gru flat e-gkn egnn; do
        echo "Running Model: $model"
        python3 -u -m koopman_evolver.cli train --md17 $mol --model $model --epochs $EPOCHS --batch-size $BATCH_SIZE
    done
    
    echo "--- Evaluating MD17: $mol ---"
    python3 -u -m koopman_evolver.cli eval --md17 $mol \
        --koopman-ckpt ./checkpoints/graph_aware_koopman_${mol}_best.pt \
        --gru-ckpt ./checkpoints/graph_aware_gru_${mol}_best.pt \
        --flat-ckpt ./checkpoints/flat_koopman_${mol}_best.pt \
        --egkn-ckpt ./checkpoints/e_gkn_${mol}_best.pt \
        --egnn-ckpt ./checkpoints/egnn_${mol}_best.pt
done

# 2. MD22 - Large Macromolecules
# We lower the batch size to 16 for MD22 to prevent VRAM spikes
MD22_BATCH=16
for mol in stachyose ac-ala3-nhme dha at-at; do
    echo "--- Training MD22: $mol ---"
    for model in koopman gru flat e-gkn egnn; do
        echo "Running Model: $model"
        python3 -u -m koopman_evolver.cli train --md22 $mol --model $model --epochs $EPOCHS --batch-size $MD22_BATCH
    done
    
    echo "--- Evaluating MD22: $mol ---"
    python3 -u -m koopman_evolver.cli eval --md22 $mol \
        --koopman-ckpt ./checkpoints/graph_aware_koopman_${mol}_best.pt \
        --gru-ckpt ./checkpoints/graph_aware_gru_${mol}_best.pt \
        --flat-ckpt ./checkpoints/flat_koopman_${mol}_best.pt \
        --egkn-ckpt ./checkpoints/e_gkn_${mol}_best.pt \
        --egnn-ckpt ./checkpoints/egnn_${mol}_best.pt
done

# 3. N-Body Physics
echo "--- Training N-Body Systems ---"
for mol in charged springs; do
    echo "--- Training N-Body: $mol ---"
    for model in e-gkn egnn koopman gru flat; do
        echo "Running Model: $model"
        python3 -u -m koopman_evolver.cli train --nbody $mol --model $model --epochs $EPOCHS --batch-size $BATCH_SIZE
    done

    echo "--- Evaluating N-Body: $mol ---"
    python3 -u -m koopman_evolver.cli eval --nbody $mol \
        --koopman-ckpt ./checkpoints/graph_aware_koopman_${mol}_best.pt \
        --gru-ckpt ./checkpoints/graph_aware_gru_${mol}_best.pt \
        --flat-ckpt ./checkpoints/flat_koopman_${mol}_best.pt \
        --egkn-ckpt ./checkpoints/e_gkn_${mol}_best.pt \
        --egnn-ckpt ./checkpoints/egnn_${mol}_best.pt
done


echo "====================================================="
echo " ALL TRAINING AND EVALUATIONS COMPLETED SUCCESSFULLY!"
echo " Checkpoints are available in ./checkpoints/"
echo "====================================================="
