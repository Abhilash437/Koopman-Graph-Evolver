import streamlit as st
import os
import argparse
from PIL import Image

# Import our CLI logic to reuse it directly in the GUI!
from koopman_evolver.cli import train, evaluate

st.set_page_config(page_title="Koopman Graph Evolver", layout="wide")

st.title("🧬 Graph-Aware Koopman Dynamics Evolver")
st.markdown("Explore deterministic, long-horizon global dynamics for MD17 molecules using structurally constrained Koopman Autoencoders vs Unconstrained GRUs.")

st.sidebar.header("Configuration")
dataset = st.sidebar.selectbox("Dataset", ["MD17", "MD22"])
if dataset == "MD17":
    molecule = st.sidebar.selectbox("Molecule", ["ethanol", "malonaldehyde", "aspirin"])
else:
    molecule = st.sidebar.selectbox("Molecule", ["ac-ala3-nhme", "dha", "at-at", "stachyose"])
model_type = st.sidebar.selectbox("Model Architecture", ["koopman", "gru", "flat"])
epochs = st.sidebar.slider("Training Epochs", min_value=1, max_value=200, value=50)
rollout_steps = st.sidebar.slider("Evaluation Rollout Steps", min_value=10, max_value=50, value=29)

st.header("1. Training Engine")
st.write("Train the selected model on the MD17 dataset. Models are automatically checkpointed.")

if st.button(f"Train {model_type.upper()} on {molecule.capitalize()}"):
    with st.spinner(f"Training {model_type} on {molecule} for {epochs} epochs... (Check your terminal for live logs!)"):
        args = argparse.Namespace(
            md17=molecule if dataset == "MD17" else None,
            md22=molecule if dataset == "MD22" else None,
            model=model_type,
            epochs=epochs,
            batch_size=16,
            hidden_dim=64,
            lr=1e-3,
            out_dir="./checkpoints"
        )
        try:
            train(args)
            st.success(f"Training complete! Checkpoint saved in `./checkpoints`.")
        except Exception as e:
            st.error(f"Error during training: {e}")

st.header("2. PhysicsEval Diagnostics")
st.write("Run deep physical diagnostics (MSE, Graph Energy, Geometry Retention) on trained checkpoints.")

koop_ckpt = f"./checkpoints/graph_aware_koopman_{molecule}_best.pt"
gru_ckpt = f"./checkpoints/graph_aware_gru_{molecule}_best.pt"
flat_ckpt = f"./checkpoints/flat_koopman_{molecule}_best.pt"

can_eval = os.path.exists(koop_ckpt) and os.path.exists(gru_ckpt)
has_flat = os.path.exists(flat_ckpt)

if can_eval:
    if has_flat:
        st.info("✅ All 3 Checkpoints found! Ready for massive 3-way ablation.")
    else:
        st.info("✅ Koopman & GRU Checkpoints found! Ready for standard evaluation. (Train 'flat' to unlock 3-way ablation).")
        
    if st.button("Run Evaluation Suite"):
        with st.spinner("Running PhysicsEval suite and generating plots..."):
            args = argparse.Namespace(
                md17=molecule if dataset == "MD17" else None,
                md22=molecule if dataset == "MD22" else None,
                koopman_ckpt=koop_ckpt,
                gru_ckpt=gru_ckpt,
                flat_ckpt=flat_ckpt if has_flat else None,
                rollout_steps=rollout_steps,
                out_dir="./results"
            )
            try:
                evaluate(args)
                st.success("Evaluation complete! Scroll down to see the results.")
            except Exception as e:
                st.error(f"Error during evaluation: {e}")
else:
    st.warning("⚠️ You must train both the Koopman and GRU models for this molecule to unlock the comparative evaluation suite.")

st.header("3. Results Dashboard")
plot_path_ablation = f"./results/ablation_{molecule}.png"
plot_path_tradeoff = f"./results/dynamics_tradeoff_{molecule}.png"

if os.path.exists(plot_path_ablation):
    st.image(Image.open(plot_path_ablation), caption=f"3-Way Phase 9 Ablation for {molecule.capitalize()}", use_container_width=True)
elif os.path.exists(plot_path_tradeoff):
    st.image(Image.open(plot_path_tradeoff), caption=f"Dynamics Tradeoff: Koopman vs GRU for {molecule.capitalize()}", use_container_width=True)
else:
    st.write("No plots generated yet. Run the evaluation suite above!")
