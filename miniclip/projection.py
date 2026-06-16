import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Linear projection into shared latent space, then L2-normalize.

    After L2-norm, ⟨zᵢ, zⱼ⟩ = cos(θᵢⱼ) ∈ [-1, 1], so dot products
    in the loss are cosine similarities on the unit hypersphere S^(d-1).
    Bias is omitted: a bias would shift every embedding off the sphere
    and the subsequent L2-norm would undo it anyway.
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.linear(x), p=2, dim=-1)
