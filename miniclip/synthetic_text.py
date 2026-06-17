import torch


class SyntheticTextPairs:
    """Synthetic image/text pairs where the text side is a token sequence.

    For each concept k ∈ [0, num_concepts):
      • Image side: x_img = A · c_k + ε  (same construction as SyntheticPairs)
      • Text side : tokens = [BOS, topic_k, topic_k, …, topic_k, EOS, PAD…]
        where topic_k is a unique token per concept.

    Special token IDs reserved at the low end:
        PAD = 0, BOS = 1, EOS = 2
    Concept tokens occupy IDs 3 … num_concepts+2, so vocab_size = num_concepts + 3.

    Intentionally simple — step 2 is verifying the Transformer machinery
    (embeddings, attention, LN, EOS pooling) end-to-end. Real language
    structure comes with Flickr8k in step 4.
    """

    PAD = 0
    BOS = 1
    EOS = 2

    def __init__(
        self,
        num_concepts: int = 256,
        d_concept: int = 32,
        d_img: int = 512,
        max_seq_len: int = 16,
        pattern_len: int = 4,
        noise_std: float = 0.3,
        seed: int = 0,
    ):
        assert pattern_len + 2 <= max_seq_len, "BOS + pattern + EOS must fit in max_seq_len"
        g = torch.Generator().manual_seed(seed)
        self.concepts = torch.randn(num_concepts, d_concept, generator=g)
        self.A = torch.randn(d_concept, d_img, generator=g) / (d_concept ** 0.5)
        self.noise_std = noise_std
        self.num_concepts = num_concepts
        self.d_img = d_img
        self.max_seq_len = max_seq_len
        self.pattern_len = pattern_len
        self.vocab_size = num_concepts + 3

    def sample_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert batch_size <= self.num_concepts, (
            f"batch_size={batch_size} must be ≤ num_concepts={self.num_concepts}"
        )
        idx = torch.randperm(self.num_concepts)[:batch_size]
        c = self.concepts[idx]
        x_img = c @ self.A + self.noise_std * torch.randn(batch_size, self.d_img)

        topic_ids = idx + 3  # offset past PAD, BOS, EOS
        tokens = torch.full((batch_size, self.max_seq_len), self.PAD, dtype=torch.long)
        tokens[:, 0] = self.BOS
        tokens[:, 1 : 1 + self.pattern_len] = topic_ids.unsqueeze(1)
        eos_pos = 1 + self.pattern_len
        tokens[:, eos_pos] = self.EOS

        eos_positions = torch.full((batch_size,), eos_pos, dtype=torch.long)
        return x_img, tokens, eos_positions
