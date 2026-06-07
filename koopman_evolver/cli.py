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
from koopman_evolver.models.baselines import GraphAwareGRUNet, FlatKoopmanNet
from koopman_evolver.training.trainer import GraphAwareTrainer
from koopman_evolver.evaluation.physics_eval import GraphAwareKoopmanEvaluator, PhysicsEval, ThreeWayAblationEvaluator

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
    group_train = train_parser.add_mutually_exclusive_group(required=True)
    group_train.add_argument("--md17", type=str, help="MD17 molecule to train on (e.g. ethanol, aspirin, malonaldehyde)")
    group_train.add_argument("--md22", type=str, help="MD22 molecule to train on (e.g. ac-ala3-nhme)")
    
    train_parser.add_argument("--model", type=str, default="koopman", choices=["koopman", "gru", "flat"], help="Model architecture")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    train_parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension size")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--out-dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    
    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model")
    group_eval = eval_parser.add_mutually_exclusive_group(required=True)
    group_eval.add_argument("--md17", type=str, help="MD17 molecule to evaluate")
    group_eval.add_argument("--md22", type=str, help="MD22 molecule to evaluate")
    
    eval_parser.add_argument("--koopman-ckpt", type=str, required=True, help="Path to Koopman model checkpoint")
    eval_parser.add_argument("--gru-ckpt", type=str, required=True, help="Path to GRU baseline checkpoint")
    eval_parser.add_argument("--flat-ckpt", type=str, default=None, help="Path to Flat Koopman checkpoint for 3-way ablation")
    eval_parser.add_argument("--rollout-steps", type=int, default=29, help="Number of steps for rollout evaluation")
    eval_parser.add_argument("--out-dir", type=str, default="./results", help="Output directory for plots")
    
    return parser

def get_data_path(dataset: str, molecule: str) -> str:
    import kagglehub
    import glob
    print(f"Downloading/Locating dataset for {dataset} - {molecule} via kagglehub...")
    if dataset == "md17":
        if molecule == "ethanol":
            path = kagglehub.dataset_download('abhilash437/md17-ethanol')
        else:
            path = kagglehub.dataset_download(f'abhilash437/rmd17-{molecule}')
    elif dataset == "md22":
        path = kagglehub.dataset_download(f'abhilash437/md22-{molecule}')
    else:
        raise ValueError(f"Unknown dataset {dataset}")
        
    npz_files = glob.glob(os.path.join(path, "*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz file found in {path}")
    return npz_files[0]

def train(args):
    device = get_device()
    dataset = "md17" if args.md17 else "md22"
    molecule = args.md17 if args.md17 else args.md22
    
    print(f"[{molecule}] Initializing {dataset} dataset on {device}...")
    
    data_path = get_data_path(dataset, molecule)
    if dataset == "md22":
        adapter = MD22Adapter(path=data_path, molecule=molecule)
    else:
        adapter = MD17AdapterV2(path=data_path, molecule=molecule)
    # The load method actually does the extraction, I need to call it!
    train_split, test_split = adapter.load()
    
    n_atoms = adapter._n_atoms # adapter determines this during load()
    edge_index = train_split.edge_index
    
    # Latent dim = n_atoms * hidden_dim
    latent_dim = n_atoms * args.hidden_dim
    
    print(f"[{molecule}] Initializing {args.model} model...")
    if args.model == "koopman":
        model = GraphAwareKoopmanNet(
            edge_index=edge_index,
            node_dim=6, edge_dim=1, hidden_dim=args.hidden_dim, 
            latent_dim=latent_dim, n_atoms=n_atoms
        )
        ckpt_name = f"graph_aware_koopman_{molecule}_best.pt"
    elif args.model == "gru":
        model = GraphAwareGRUNet(
            edge_index=edge_index,
            node_dim=6, edge_dim=1, hidden_dim=args.hidden_dim, 
            latent_dim=latent_dim, n_atoms=n_atoms
        )
        ckpt_name = f"graph_aware_gru_{molecule}_best.pt"
    elif args.model == "flat":
        flat_latent = 42 * args.hidden_dim if molecule == "stachyose" else latent_dim
        model = FlatKoopmanNet(
            n_atoms=n_atoms,
            input_dim=6,
            latent_dim=flat_latent
        )
        ckpt_name = f"flat_koopman_{molecule}_best.pt"
        
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
    
    print(f"[{molecule}] Starting training...")
    trainer.fit(train_split, test_split)
    print(f"[{molecule}] Training complete. Best checkpoint saved to {os.path.join(args.out_dir, ckpt_name)}")

def evaluate(args):
    device = get_device()
    dataset = "md17" if args.md17 else "md22"
    molecule = args.md17 if args.md17 else args.md22
    
    print(f"[{molecule}] Initializing {dataset} dataset for evaluation...")
    
    data_path = get_data_path(dataset, molecule)
    if dataset == "md22":
        adapter = MD22Adapter(path=data_path, molecule=molecule)
    else:
        adapter = MD17AdapterV2(path=data_path, molecule=molecule)
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
    
    if args.flat_ckpt and os.path.exists(args.flat_ckpt):
        print("\nRunning Massive 3-Way Ablation Evaluation...")
        flat_latent = 42 * 64 if molecule == "stachyose" else latent_dim
        flat_model = FlatKoopmanNet(n_atoms=n_atoms, input_dim=6, latent_dim=flat_latent)
        f_ckpt = torch.load(args.flat_ckpt, map_location=device, weights_only=False)
        flat_model.load_state_dict(f_ckpt["model_state_dict"])
        print(f"Loaded Flat Koopman checkpoint from epoch {f_ckpt['epoch']}")
        
        evaluator = ThreeWayAblationEvaluator(
            flat_model=flat_model,
            graph_koop_model=koopman_model,
            graph_gru_model=gru_model,
            device=device,
            rollout_steps=args.rollout_steps,
            n_atoms=n_atoms
        )
        results = evaluator.run(test_split, steps=args.rollout_steps)
        evaluator.print_summary(results)
        evaluator.plot(results, save_path=os.path.join(args.out_dir, f"ablation_{molecule}.png"))
        
    else:
        print("\nRunning 2-Way Evaluation Suite...")
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
        evaluator.plot(results, save_path=os.path.join(args.out_dir, f"dynamics_tradeoff_{molecule}.png"))
        
        print("\nRunning Deep Physical Diagnostics (Bonds/Angles/Torsions)...")
        physics_eval = PhysicsEval(koopman_model, gru_model, test_split, n_atoms, molecule)
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
