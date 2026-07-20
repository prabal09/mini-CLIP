"""Step 4b — distributed mini-CLIP training with DDP + all-gathered negatives.

Launch:
    torchrun --nproc_per_node=<N> train_flickr8k_ddp.py
    (N=1 is a valid smoke-test — the DDP code paths still execute.)

Backend:
    NCCL when CUDA is available and built with NCCL support (Linux, multi-GPU).
    gloo elsewhere (Windows, CPU, single-GPU CUDA without NCCL). Picked
    automatically.

The design decision beyond wrapping the model in DDP is in the loss:
naive DDP + InfoNCE undertrains. Each rank only computes InfoNCE against
its LOCAL batch's negatives, so the effective negative pool stays at
`local_batch` regardless of world_size — you get the throughput of
scaling but not the contrastive benefit of a larger batch. To fix this,
we all-gather the projected embeddings across ranks BEFORE computing
the loss, so every rank operates on a (global_batch, global_batch)
similarity matrix. Global batch = per-rank batch × world_size.

Gradients flow through the local shard only (standard OpenCLIP trick:
replace the local slot of the gathered list with the local tensor that
carries a grad_fn). Cross-rank gradient combination is handled by DDP's
default all-reduce on model parameters — each rank contributes gradient
for its own encoder's local shard, and DDP averages across ranks.
"""

import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from miniclip import (
    ProjectionHead,
    LearnedTemperature,
    symmetric_infonce_loss,
    TextTransformer,
    VisionTransformer,
    SimpleWordTokenizer,
    Flickr8kDataset,
    collate_fn,
    make_train_transform,
    make_eval_transform,
    compute_retrieval_metrics,
    AMPContext,
)


# ---------------- distributed helpers ----------------

def setup_dist():
    """Initialize the process group. Reads env vars set by torchrun."""
    backend = "nccl" if (torch.cuda.is_available() and dist.is_nccl_available()) else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size(), backend


def is_main(rank: int) -> bool:
    return rank == 0


def log(rank: int, *args, **kwargs):
    """Rank-0-only print — avoids interleaved output from every process."""
    if is_main(rank):
        print(*args, **kwargs)


def gather_with_grad(local: torch.Tensor, world_size: int) -> torch.Tensor:
    """All-gather local embeddings, keeping the local slot's grad_fn.

    dist.all_gather returns DETACHED tensors — gradients to the gathered
    non-local shards are lost. Replacing gathered[rank] with `local`
    (which still has grad_fn) means the loss's gradient w.r.t. the local
    shard flows back into the local encoder correctly. Other ranks handle
    their own shards symmetrically; DDP's parameter all-reduce combines
    them.
    """
    if world_size == 1:
        return local
    gathered = [torch.zeros_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    gathered[dist.get_rank()] = local
    return torch.cat(gathered, dim=0)


# ---------------- CLIP wrapper module ----------------

class CLIPModule(torch.nn.Module):
    """Bundles the five submodules so a single DDP wrap covers everything.

    Wrapping each encoder/head/temperature in its own DDP handle also
    works but adds five backward hooks and five sets of gradient buckets.
    One wrapper = one hook.
    """

    def __init__(self, image_encoder, text_encoder, proj_img, proj_text, temperature, amp_ctx):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.proj_img = proj_img
        self.proj_text = proj_text
        self.temperature = temperature
        self.amp_ctx = amp_ctx

    def forward(self, images, tokens, eos_pos):
        # Encoders inside autocast (heavy matmuls use FP16 tensor cores).
        # Projection heads in FP32 so L2-norm + similarity stay safe.
        with self.amp_ctx.autocast():
            img_feat = self.image_encoder(images)
            text_feat = self.text_encoder(tokens, eos_pos)
        z_img = self.proj_img(img_feat.float())
        z_text = self.proj_text(text_feat.float())
        return z_img, z_text, self.temperature()


def main():
    # ---------------- distributed init ----------------
    local_rank, rank, world_size, backend = setup_dist()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    log(rank, f"Distributed: rank {rank}/{world_size} | backend={backend} | device={device}")

    # ---------------- config ----------------
    DATA_ROOT = "./flickr8k"
    IMAGE_SIZE = 96
    PATCH_SIZE = 8
    MAX_SEQ_LEN = 32
    D_VIT = 256
    D_TEXT = 256
    D_LATENT = 128
    NUM_HEADS = 4
    NUM_LAYERS = 4
    D_FF = 1024
    BATCH_SIZE = 64                  # PER-RANK. Global batch = BATCH_SIZE * world_size.
    EPOCHS = 20
    LR = 5e-4
    WEIGHT_DECAY = 0.1
    GRAD_CLIP = 1.0
    VAL_SPLIT = 0.2
    MIN_WORD_FREQ = 3
    NUM_WORKERS = 2
    USE_AMP = True
    AMP_DTYPE = torch.float16
    SEED = 0

    amp_ctx = AMPContext(enabled=USE_AMP, device=device, dtype=AMP_DTYPE)
    log(rank, f"AMP: {amp_ctx.enabled} | Global batch: {BATCH_SIZE * world_size}")

    # Same seed on every rank so vocab + train/val split are identical
    # everywhere. DistributedSampler handles per-rank shuffling separately.
    torch.manual_seed(SEED)
    random.seed(SEED)

    # ---------------- tokenizer ----------------
    log(rank, "Building tokenizer …")
    captions = Flickr8kDataset.all_captions(DATA_ROOT)
    tokenizer = SimpleWordTokenizer()
    tokenizer.build_vocab(captions, min_freq=MIN_WORD_FREQ)
    log(rank, f"Captions: {len(captions):,} | Vocab: {tokenizer.vocab_size:,}")

    # ---------------- datasets ----------------
    train_ds_full = Flickr8kDataset(
        DATA_ROOT, tokenizer, MAX_SEQ_LEN,
        image_transform=make_train_transform(IMAGE_SIZE),
    )
    val_ds_full = Flickr8kDataset(
        DATA_ROOT, tokenizer, MAX_SEQ_LEN,
        image_transform=make_eval_transform(IMAGE_SIZE),
    )

    image_names = sorted({name for name, _ in train_ds_full.samples})
    rng = random.Random(SEED)
    rng.shuffle(image_names)
    n_val_images = int(len(image_names) * VAL_SPLIT)
    val_images = set(image_names[:n_val_images])

    train_indices = [i for i, (n, _) in enumerate(train_ds_full.samples) if n not in val_images]
    val_indices = [i for i, (n, _) in enumerate(val_ds_full.samples) if n in val_images]
    train_ds = Subset(train_ds_full, train_indices)
    val_ds = Subset(val_ds_full, val_indices)
    log(rank, f"Train pairs: {len(train_ds):,} | Val pairs: {len(val_ds):,}")

    # DistributedSampler: each rank sees a disjoint slice of the dataset.
    # `set_epoch(epoch)` must be called each epoch or every epoch shuffles
    # the same way — a common silent bug in distributed training.
    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank,
        shuffle=True, seed=SEED, drop_last=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, drop_last=True,
    )
    # Validation runs on rank 0 only (small dataset, retrieval metrics
    # are inherently global and awkward to reduce piecewise).
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
    )

    # ---------------- model ----------------
    image_encoder = VisionTransformer(
        image_size=IMAGE_SIZE, patch_size=PATCH_SIZE,
        d_model=D_VIT, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, d_ff=D_FF,
    ).to(device)
    text_encoder = TextTransformer(
        vocab_size=tokenizer.vocab_size, max_seq_len=MAX_SEQ_LEN,
        d_model=D_TEXT, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, d_ff=D_FF,
    ).to(device)
    proj_img = ProjectionHead(D_VIT, D_LATENT).to(device)
    proj_text = ProjectionHead(D_TEXT, D_LATENT).to(device)
    temperature = LearnedTemperature(init_tau=0.07).to(device)

    module = CLIPModule(image_encoder, text_encoder, proj_img, proj_text, temperature, amp_ctx)
    # device_ids is required for NCCL; gloo/CPU takes no device_ids.
    ddp_kwargs = {"device_ids": [local_rank]} if torch.cuda.is_available() else {}
    model = DDP(module, **ddp_kwargs)

    if is_main(rank):
        n_params = sum(p.numel() for p in model.parameters())
        log(rank, f"Trainable params: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * EPOCHS
    )

    # ---------------- training loop ----------------
    for epoch in range(EPOCHS):
        # Without this, all epochs shuffle to the same order.
        train_sampler.set_epoch(epoch)
        model.train()

        running = {"loss": 0.0, "acc_i2t": 0.0, "acc_t2i": 0.0}
        for step, (images, tokens, eos_pos) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            eos_pos = eos_pos.to(device, non_blocking=True)

            z_img, z_text, temp = model(images, tokens, eos_pos)

            # All-gather AFTER projection: every rank now holds
            # (world_size * BATCH_SIZE, D_LATENT) worth of negatives.
            z_img_all = gather_with_grad(z_img, world_size)
            z_text_all = gather_with_grad(z_text, world_size)
            loss, m = symmetric_infonce_loss(z_img_all, z_text_all, temp)

            optimizer.zero_grad()
            amp_ctx.backward(loss)
            # Unscale grads BEFORE clipping so the clip threshold applies
            # to true-magnitude gradients, not loss-scaled ones.
            amp_ctx.unscale(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            amp_ctx.step(optimizer)
            scheduler.step()
            # In-place clamp on the local tensor. All ranks see the same
            # gradients (via DDP) so logit_scale stays in sync without
            # explicit broadcast.
            model.module.temperature.clamp_()

            for k in running:
                running[k] += m[k]

            if step % 50 == 0:
                log(rank,
                    f"epoch {epoch} step {step:>4}/{len(train_loader)} | "
                    f"loss {m['loss']:.4f} | acc_i2t {m['acc_i2t']:.3f} | "
                    f"τ {model.module.temperature.tau():.4f} | "
                    f"lr {scheduler.get_last_lr()[0]:.2e}"
                )

        n_steps = len(train_loader)
        log(rank,
            f"==> epoch {epoch} done | "
            f"avg loss {running['loss']/n_steps:.4f} | "
            f"avg acc_i2t {running['acc_i2t']/n_steps:.3f}"
        )

        # Validation on rank 0. Others wait at the barrier so the next
        # epoch's first all_gather doesn't fire before rank 0 catches up.
        if is_main(rank):
            metrics = compute_retrieval_metrics(
                model.module.image_encoder, model.module.text_encoder,
                model.module.proj_img, model.module.proj_text, val_loader, device,
            )
            log(rank, "    val: " + " | ".join(f"{k} {v:.3f}" for k, v in metrics.items()))
        dist.barrier()

    # ---------------- save (rank 0 only) ----------------
    if is_main(rank):
        Path("./checkpoints").mkdir(exist_ok=True)
        m = model.module
        torch.save({
            "image_encoder": m.image_encoder.state_dict(),
            "text_encoder": m.text_encoder.state_dict(),
            "proj_img": m.proj_img.state_dict(),
            "proj_text": m.proj_text.state_dict(),
            "logit_scale": m.temperature.logit_scale.detach().cpu(),
            "tokenizer_vocab": tokenizer.id_to_word,
            "config": {
                "IMAGE_SIZE": IMAGE_SIZE, "PATCH_SIZE": PATCH_SIZE,
                "MAX_SEQ_LEN": MAX_SEQ_LEN, "D_VIT": D_VIT, "D_TEXT": D_TEXT,
                "D_LATENT": D_LATENT, "NUM_HEADS": NUM_HEADS,
                "NUM_LAYERS": NUM_LAYERS, "D_FF": D_FF,
            },
        }, "./checkpoints/mini_clip_flickr8k_ddp.pt")
        log(rank, "Saved checkpoint to ./checkpoints/mini_clip_flickr8k_ddp.pt")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
