import torch


class SyntheticPairs:
    """Synthetic image/text feature pairs that share latent 'concepts'.

    Construction:
      • K concept vectors c_k ∈ ℝ^d_concept are drawn once and frozen.
      • Two random linear maps A: d_concept → d_img and B: d_concept → d_text
        simulate two different "encoders" reading the same concept.
      • Each pair samples a concept index k and produces:

            x_img  = A(c_k) + ε_img
            x_text = B(c_k) + ε_text

    Matched pairs share underlying structure (the same c_k), so a
    contrastive model that learns to project them into a shared space
    can recover that structure. This is the controlled microcosm for
    testing the loss + temperature before real encoders exist.

    A and B are scaled by 1/√d_concept so x_img / x_text have roughly
    unit variance per dimension, which keeps the noise floor meaningful.
    """

    def __init__(
        self,
        num_concepts: int = 256,
        d_concept: int = 32,
        d_img: int = 512,
        d_text: int = 384,
        noise_std: float = 0.3,
        seed: int = 0,
    ):
        g = torch.Generator().manual_seed(seed)
        self.concepts = torch.randn(num_concepts, d_concept, generator=g)
        self.A = torch.randn(d_concept, d_img, generator=g) / (d_concept ** 0.5)
        self.B = torch.randn(d_concept, d_text, generator=g) / (d_concept ** 0.5)
        self.noise_std = noise_std
        self.num_concepts = num_concepts
        self.d_img = d_img
        self.d_text = d_text

    def sample_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Distinct concepts per batch so positives are genuinely different —
        # if two rows shared a concept, the contrastive loss would penalize
        # the model for treating them as similar, which is wrong.
        assert batch_size <= self.num_concepts, (
            f"batch_size={batch_size} must be ≤ num_concepts={self.num_concepts}"
        )
        idx = torch.randperm(self.num_concepts)[:batch_size]
        c = self.concepts[idx]
        x_img = c @ self.A + self.noise_std * torch.randn(batch_size, self.d_img)
        x_text = c @ self.B + self.noise_std * torch.randn(batch_size, self.d_text)
        return x_img, x_text
