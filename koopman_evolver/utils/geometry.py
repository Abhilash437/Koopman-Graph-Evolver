import torch

def safe_matrix_exp(A: torch.Tensor) -> torch.Tensor:
    """
    Computes matrix exponential K = exp(A).
    Safely falls back to CPU if tensor is on Apple Silicon MPS device,
    as PyTorch MPS lacks native `aten::linalg_matrix_exp` implementation.
    """
    if A.device.type == "mps":
        return torch.matrix_exp(A.cpu()).to(A.device)
    return torch.matrix_exp(A)
