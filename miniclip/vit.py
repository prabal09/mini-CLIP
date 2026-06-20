import torch
import torch.nn as nn

from .transformer import TransformerBlock


class PatchEmbed(nn.Module):
    """Patchify image + linear-project each patch to d_model.

    A Conv2d with kernel_size=P, stride=P applied to (B, 3, H, W) is
    mathematically equivalent to:
      • extract non-overlapping P×P patches  → (B, N, P²·3)
      • multiply by a (P²·3, d_model) matrix → (B, N, d_model)
    The Conv2d form is one operation and what the original ViT uses.

    N = (H / P) × (W / P) is the number of patches.
    """

    def __init__(self, image_size: int, patch_size: int, in_channels: int, d_model: int):
        super().__init__()
        assert image_size % patch_size == 0, "patch_size must divide image_size"
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                # (B, d_model, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, d_model)
        return x


class VisionTransformer(nn.Module):
    """Tiny ViT for the image side of mini-CLIP.

    Pipeline:
        image → patchify + linear-project        (B, N, d_model)
              → prepend learnable CLS token      (B, 1+N, d_model)
              → add learned positional embed     (B, 1+N, d_model)
              → K × TransformerBlock (non-causal)
              → final LayerNorm
              → pool: hidden state at position 0 (the CLS slot)

    Why CLS pooling? With non-causal attention, every position attends
    to every other position in every layer. The CLS token has no
    intrinsic spatial meaning — it's a designated readout slot that the
    model can fill with whatever summary minimizes the contrastive loss.

    Why learned positional embeddings rather than sinusoidal? Patchify
    + linear destroys spatial information; positional embeddings restore
    it. Learned tables are simpler and what CLIP / original ViT use.
    Sinusoidal would generalize to other image sizes but isn't needed here.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int = 3,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 1024,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, d_model)
        N = self.patch_embed.num_patches

        # CLS is a (1, 1, d_model) parameter that is broadcast across the
        # batch each forward pass. Stored with batch dim 1 so torch.cat
        # is one line.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # One learnable position per slot in the (CLS + N patches) sequence.
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + N, d_model))
        # Small-normal init is the ViT-paper default; large values would
        # dominate the patch embeddings at start.
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, causal=False)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, C, H, W). Returns (B, d_model) image embeddings."""
        B = images.shape[0]
        x = self.patch_embed(images)                      # (B, N, d_model)
        cls = self.cls_token.expand(B, -1, -1)            # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                    # (B, 1+N, d_model)
        x = x + self.pos_embed                            # broadcast (1, 1+N, d) across batch

        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)

        return x[:, 0]                                    # CLS pool
