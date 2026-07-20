import contextlib

import torch
from torch.amp import GradScaler, autocast


class AMPContext:
    """Mixed-precision training helper.

    Wraps three pieces of PyTorch's AMP API behind one object so the
    training loop can call them uniformly whether AMP is enabled or not:

      • autocast() — context manager for FP16/BF16 forward passes
      • backward(loss) — scaled backward when enabled, plain otherwise
      • unscale(opt) — undo loss-scale on grads BEFORE clipping
      • step(opt) — optimizer step + scaler bookkeeping

    Why is AMP CUDA-only here? On CPU, FP16 ops generally run slower
    than FP32 (no tensor cores; the dtype-conversion overhead dominates).
    BF16 on newer Intel/AMD chips can help, but isn't universally
    available — we gate on `device.type == "cuda"` for safety.

    Why the contrastive loss must stay in FP32:
        logits = sim · exp(t),    sim ∈ [-1, 1],    exp(t) up to 100
        ⇒ logits ∈ [-100, 100]
    FP16's representable range tops out at ~65504, but the LSE inside
    cross_entropy computes log Σ exp(logits) — exp(100) ≈ 2.7e43, which
    overflows even before max-subtraction kicks in if intermediate
    storage is FP16. Real CLIP keeps the loss in FP32 for this reason.

    Usage in a training step:
        with amp_ctx.autocast():
            features = encoder(x)               # FP16 (fast)
        z = projection_head(features.float())   # FP32 (safe)
        loss = contrastive_loss(z, ...)         # FP32 (safe)
        optimizer.zero_grad()
        amp_ctx.backward(loss)
        amp_ctx.unscale(optimizer)              # so clip threshold is meaningful
        clip_grad_norm_(params, ...)
        amp_ctx.step(optimizer)
    """

    def __init__(
        self,
        enabled: bool,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        # Only CUDA actually benefits from AMP in practice.
        self.enabled = bool(enabled) and device.type == "cuda"
        self.device_type = device.type
        self.dtype = dtype
        # GradScaler accepts enabled=False; in that mode every method
        # is a passthrough, but we still avoid creating the underlying
        # state when AMP is off.
        self.scaler = GradScaler(device.type, enabled=self.enabled)

    def autocast(self):
        if not self.enabled:
            return contextlib.nullcontext()
        return autocast(device_type=self.device_type, dtype=self.dtype)

    def backward(self, loss: torch.Tensor) -> None:
        if self.enabled:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def unscale(self, optimizer: torch.optim.Optimizer) -> None:
        # Must be called before grad clipping when AMP is on, so that
        # the clip threshold is applied to true-magnitude gradients.
        if self.enabled:
            self.scaler.unscale_(optimizer)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.enabled:
            # scaler.step skips the optimizer update if inf/nan grads
            # were detected, and scaler.update tunes the loss scale.
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
