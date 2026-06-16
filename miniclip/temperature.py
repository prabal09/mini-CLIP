import math
import torch
import torch.nn as nn


class LearnedTemperature(nn.Module):
    """Learnable inverse-temperature for the contrastive loss.

    We parameterize t = log(1/τ) — the "logit scale" — as a trainable scalar.
    Forward returns exp(t), which multiplies the cosine similarities:

        logits = similarity · exp(t)   ⇔   logits = similarity / τ

    Two reasons for storing t instead of τ directly:
      1. Positivity for free:  τ = exp(-t) > 0 for any real t.
      2. Multiplicative-space updates: gradients on t behave well across
         orders of magnitude of τ (e.g., τ = 0.5 vs τ = 0.01).

    Clamp protects against τ → 0 collapse, which would create exploding
    gradients. CLIP's choice (max_logit_scale=100, i.e. τ ≥ 0.01) is the
    default. Clamping is done in-place on the parameter after each
    optimizer step so the parameter never drifts past the limit.
    """

    def __init__(self, init_tau: float = 0.07, max_logit_scale: float = 100.0):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_tau)))
        self.max_logit_scale = math.log(max_logit_scale)

    def forward(self) -> torch.Tensor:
        return self.logit_scale.exp()

    @torch.no_grad()
    def clamp_(self) -> None:
        self.logit_scale.clamp_(max=self.max_logit_scale)

    @torch.no_grad()
    def tau(self) -> float:
        return (1.0 / self.forward()).item()
