import torch
import torch.nn as nn

from .transformer import TransformerBlock


class TextTransformer(nn.Module):
    """Tiny causal Transformer text encoder, CLIP-style.

    Pipeline:
        tokens → token_embed + pos_embed (both learned via nn.Embedding)
               → N × TransformerBlock (causal self-attention)
               → final LayerNorm
               → pool: hidden state at EOS position

    Causal masking + EOS pooling is the design choice that gives a
    well-defined sentence vector: with the mask, only the EOS position
    has attended to the entire sequence, so its hidden state is the
    canonical summary of the caption.

    Why ADD token + position embeddings rather than concatenate?
    Concatenation would double d_model and require a projection to
    fold them back. Addition lets the network learn to disentangle the
    two signals across the feature dimensions — a free, compact design
    that has held up across years of Transformer variants.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 1024,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, causal=True)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.max_seq_len = max_seq_len

    def forward(self, tokens: torch.Tensor, eos_positions: torch.Tensor) -> torch.Tensor:
        """
        tokens:        (B, T) long
        eos_positions: (B,)   long — index of EOS in each row

        Returns sentence embeddings (B, d_model).
        """
        B, T = tokens.shape
        positions = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)

        x = self.token_embed(tokens) + self.pos_embed(positions)
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)

        batch_idx = torch.arange(B, device=tokens.device)
        return x[batch_idx, eos_positions]
