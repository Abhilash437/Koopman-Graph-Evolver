import argparse
import os
import torch
from pathlib import Path

# Fix local imports if run as a script or via -m
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from koopman_evolver.data.md17_adapter import MD17AdapterV2
from koopman_evolver.data.md22_adapter import MD22Adapter
from koopman_evolver.data.dataset_split import GraphDatasetSplit
from koopman_evolver.models.koopman_net import GraphAwareKoopmanNet
from koopman_evolver.models.baselines import GraphAwareGRUNet
from koopman_evolver.training.trainer import GraphAwareTrainer
from koopman_evolver.evaluation.physics_eval import GraphAwareKoopmanEvaluator, PhysicsEval

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def build_parser():
    parser = argparse.ArgumentParser(description="Koopman Graph Evolver CLI")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model on MD17/MD22")
    train_parser.add_argument("--molecule", type=str, default="ethanol", choices=["ethanol", "malonaldehyde", "aspirin", "ac-ala3-nhme"], help="Molecule to train on")
    train_parser.add_argument("--model", type=str, default="koopman", choices=["koopman", "gru"], help="Model architecture")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    train_parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension size")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--out-dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    
    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model")
    eval_parser.add_argument("--molecule", type=str, default="ethanol", choices=["ethanol", "malonaldehyde", "aspirin", "ac-ala3-nhme"], help="Molecule to evaluate")
    eval_parser.add_argument("--koopman-ckpt", type=str, required=True, help="Path to Koopman model checkpoint")
    eval_parser.add_argument("--gru-ckpt", type=str, required=True, help="Path to GRU baseline checkpoint")
    eval_parser.add_argument("--rollout-steps", type=int, default=29, help="Number of steps for rollout evaluation")
    eval_parser.add_argument("--out-dir", type=str, default="./results", help="Output directory for plots")
    
    return parser

def get_data_path(molecule: str) -> str:
    import kagglehub
    import glob
    print(f"Downloading/Locating dataset for {molecule} via kagglehub...")
    if molecule == "ethanol":
        path = kagglehub.dataset_download('abhilash437/md17-ethanol')
    elif molecule == "aspirin":
        path = kagglehub.dataset_download('abhilash437/rmd17-aspirin')
    elif molecule == "malonaldehyde":
        path = kagglehub.dataset_download('abhilash437/rmd17-malonaldehyde')
    elif molecule == "ac-ala3-nhme":
        path = kagglehub.dataset_download('abhilash437/md22-ac-ala3-nhme')
    else:
        raise ValueError(f"Unknown molecule {molecule}")
        
    npz_files = glob.glob(os.path.join(path, "*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz file found in {path}")
    return npz_files[0]

def train(args):
    device = get_device()
    print(f"[{args.molecule}] Initializing dataset on {device}...")
    
    data_path = get_data_path(args.molecule)
    if args.molecule == "ac-ala3-nhme":
        adapter = MD22Adapter(path=data_path, molecule=args.molecule)
    else:
        adapter = MD17AdapterV2(path=data_path, molecule=args.molecule)
    # The load method actually does the extraction, I need to call it!
    train_split, test_split = adapter.load()
    
    n_atoms = adapter._n_atoms # adapter determines this during load()
    edge_index = train_split.edge_index
    
    # Latent dim = n_atoms * hidden_dim
    latent_dim = n_atoms * args.hidden_dim
    
    print(f"[{args.molecule}] Initializing {args.model} model...")
    if args.model == "koopman":
        model = GraphAwareKoopmanNet(
            edge_index=edge_index,
            node_dim=6, edge_dim=1, hidden_dim=args.hidden_dim, 
            latent_dim=latent_dim, n_atoms=n_atoms
        )
        ckpt_name = f"graph_aware_koopman_{args.molecule}_best.pt"
    else:
        model = GraphAwareGRUNet(
            edge_index=edge_index,
            node_dim=6, edge_dim=1, hidden_dim=args.hidden_dim, 
            latent_dim=latent_dim, n_atoms=n_atoms
        )
        ckpt_name = f"graph_aware_gru_{args.molecule}_best.pt"
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    trainer = GraphAwareTrainer(
        model=model,
        optimizer=optimizer,
        checkpoint_dir=args.out_dir,
        checkpoint_name=ckpt_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        log_every=10,
    )
    
    print(f"[{args.molecule}] Starting training...")
    trainer.fit(train_split, test_split)
    print(f"[{args.molecule}] Training complete. Best checkpoint saved to {os.path.join(args.out_dir, ckpt_name)}")

def evaluate(args):
    device = get_device()
    print(f"[{args.molecule}] Initializing dataset for evaluation...")
    
    data_path = get_data_path(args.molecule)
    if args.molecule == "ac-ala3-nhme":
        adapter = MD22Adapter(path=data_path, molecule=args.molecule)
    else:
        adapter = MD17AdapterV2(path=data_path, molecule=args.molecule)
    train_split, test_split = adapter.load()
    
    n_atoms = adapter._n_atoms
    edge_index = test_split.edge_index
    latent_dim = n_atoms * 64
    
    print("Loading models...")
    koopman_model = GraphAwareKoopmanNet(edge_index=edge_index, node_dim=6, edge_dim=1, hidden_dim=64, latent_dim=latent_dim, n_atoms=n_atoms)
    gru_model = GraphAwareGRUNet(edge_index=edge_index, node_dim=6, edge_dim=1, hidden_dim=64, latent_dim=latent_dim, n_atoms=n_atoms)
    
    k_ckpt = torch.load(args.koopman_ckpt, map_location=device, weights_only=False)
    koopman_model.load_state_dict(k_ckpt["model_state_dict"])
    print(f"Loaded Koopman checkpoint from epoch {k_ckpt['epoch']}")
    
    g_ckpt = torch.load(args.gru_ckpt, map_location=device, weights_only=False)
    gru_model.load_state_dict(g_ckpt["model_state_dict"])
    print(f"Loaded GRU checkpoint from epoch {g_ckpt['epoch']}")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("\nRunning Evaluation Suite...")
    evaluator = GraphAwareKoopmanEvaluator(
        koopman_model=koopman_model,
        baseline_model=gru_model,
        device=device,
        rollout_steps=args.rollout_steps,
        batch_size=64,
        n_atoms=n_atoms
    )
    
    results = evaluator.run(test_split)
    evaluator.print_summary(results)
    evaluator.plot(results, save_path=os.path.join(args.out_dir, f"dynamics_tradeoff_{args.molecule}.png"))
    
    print("\nRunning Deep Physical Diagnostics (Bonds/Angles/Torsions)...")
    physics_eval = PhysicsEval(koopman_model, gru_model, test_split, n_atoms, args.molecule)
    physics_eval.run(steps=args.rollout_steps, out_dir=args.out_dir)
    print(f"Evaluation complete. Plots saved to {args.out_dir}/")

def main():
    parser = build_parser()
    args = parser.parse_args()
    
    if args.command == "train":
        train(args)
    elif args.command == "eval":
        evaluate(args)

if __name__ == "__main__":
    main()
