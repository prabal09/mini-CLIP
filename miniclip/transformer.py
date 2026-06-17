import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional causal masking.

    For each head:
        Q = X·W_Q, K = X·W_K, V = X·W_V
        A = Q·Kᵀ / √d_head      ← scaling keeps Var(A) ≈ 1 across head sizes
        P = softmax(A)
        head_out = P · V
    Heads run in parallel on disjoint d_head-sized subspaces of d_model;
    outputs are concatenated and projected by W_O.

    Causal mask sets Aᵢⱼ = -∞ for j > i, so softmax assigns zero weight
    to future tokens — position i can only see positions ≤ i.
    """

    def __init__(self, d_model: int, num_heads: int, causal: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.causal = causal

        # Combined Q/K/V projection — one matmul instead of three.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        qkv = self.qkv(x)                                  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)                     # each (B, T, D)
        # Reshape to (B, h, T, d_head) so heads compute in parallel.
        q = q.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, h, T, T)

        if self.causal:
            # Upper triangle (above diagonal) = future positions.
            mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(mask, float("-inf"))

        attn = scores.softmax(dim=-1)
        out = attn @ v                                     # (B, h, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block.

        x ← x + Attn(LN(x))
        x ← x + FFN(LN(x))

    The residual stream stays unnormalized; each sublayer reads a
    normalized view of it. More stable at depth than post-LN, which is
    why GPT-2 / modern CLIP variants use this layout.

    FFN is the standard Linear → GELU → Linear with hidden width d_ff
    (commonly 4·d_model). It operates on each token position independently
    — its job is per-token feature transformation, not mixing across tokens.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, causal: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, causal=causal)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
