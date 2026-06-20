import torch


class SyntheticImageTokens:
    """Synthetic image + caption pairs sharing latent concepts.

    Each concept k gets:
      • A unique top-half RGB color    top_k    ∈ [-1, 1]^3
      • A unique bottom-half RGB color bottom_k ∈ [-1, 1]^3
      • A unique topic token (concept index + 3)

    Image (3, H, H): top half filled with top_k, bottom half with bottom_k,
                     Gaussian noise added everywhere.
    Tokens (T,):     [BOS, topic_k, topic_k, …, topic_k, EOS, PAD…]

    Two colors per concept means the ViT must combine information from
    both halves — average-pooling alone won't disambiguate concepts that
    share their average color but differ in which half is which.

    Special token IDs: PAD=0, BOS=1, EOS=2; concept topics start at 3.
    """

    PAD = 0
    BOS = 1
    EOS = 2

    def __init__(
        self,
        num_concepts: int = 256,
        image_size: int = 32,
        max_seq_len: int = 16,
        pattern_len: int = 4,
        noise_std: float = 0.3,
        seed: int = 0,
    ):
        assert pattern_len + 2 <= max_seq_len
        assert image_size % 2 == 0, "image_size must be even (top/bottom split)"
        g = torch.Generator().manual_seed(seed)
        # Colors in roughly [-1, 1]; noise will be added on top.
        self.top_colors = torch.randn(num_concepts, 3, generator=g) * 0.5
        self.bottom_colors = torch.randn(num_concepts, 3, generator=g) * 0.5

        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.pattern_len = pattern_len
        self.noise_std = noise_std
        self.num_concepts = num_concepts
        self.vocab_size = num_concepts + 3

    def sample_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert batch_size <= self.num_concepts
        idx = torch.randperm(self.num_concepts)[:batch_size]

        H = self.image_size
        images = torch.empty(batch_size, 3, H, H)
        top = self.top_colors[idx].view(batch_size, 3, 1, 1)
        bot = self.bottom_colors[idx].view(batch_size, 3, 1, 1)
        images[:, :, : H // 2, :] = top
        images[:, :, H // 2 :, :] = bot
        images = images + self.noise_std * torch.randn_like(images)

        topic_ids = idx + 3
        tokens = torch.full((batch_size, self.max_seq_len), self.PAD, dtype=torch.long)
        tokens[:, 0] = self.BOS
        tokens[:, 1 : 1 + self.pattern_len] = topic_ids.unsqueeze(1)
        eos_pos = 1 + self.pattern_len
        tokens[:, eos_pos] = self.EOS
        eos_positions = torch.full((batch_size,), eos_pos, dtype=torch.long)

        return images, tokens, eos_positions
