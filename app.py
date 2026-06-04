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
molecule = st.sidebar.selectbox("Molecule", ["ethanol", "malonaldehyde", "aspirin"])
model_type = st.sidebar.selectbox("Model Architecture", ["koopman", "gru"])
epochs = st.sidebar.slider("Training Epochs", min_value=1, max_value=200, value=50)
rollout_steps = st.sidebar.slider("Evaluation Rollout Steps", min_value=10, max_value=50, value=29)

st.header("1. Training Engine")
st.write("Train the selected model on the MD17 dataset. Models are automatically checkpointed.")

if st.button(f"Train {model_type.upper()} on {molecule.capitalize()}"):
    with st.spinner(f"Training {model_type} on {molecule} for {epochs} epochs... (Check your terminal for live logs!)"):
        args = argparse.Namespace(
            molecule=molecule,
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
can_eval = os.path.exists(koop_ckpt) and os.path.exists(gru_ckpt)

if can_eval:
    st.info("✅ Checkpoints found! Ready for evaluation.")
    if st.button("Run Evaluation Suite"):
        with st.spinner("Running PhysicsEval suite and generating plots..."):
            args = argparse.Namespace(
                molecule=molecule,
                koopman_ckpt=koop_ckpt,
                gru_ckpt=gru_ckpt,
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
plot_path = f"./results/dynamics_tradeoff_{molecule}.png"
if os.path.exists(plot_path):
    st.image(Image.open(plot_path), caption=f"Dynamics Tradeoff: Koopman vs GRU for {molecule.capitalize()}", use_container_width=True)
else:
    st.write("No plots generated yet. Run the evaluation suite above!")
