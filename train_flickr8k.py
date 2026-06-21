"""Step 4 — full mini-CLIP training on Flickr8k.

Pipeline:
  1. Build word-level tokenizer from training captions.
  2. Wrap (image, caption) pairs in a Dataset with train-time augmentations.
  3. Split 80/20 train/val BY IMAGE so val captions don't leak from
     training images.
  4. Train ViT + TextTransformer + projection heads + learned τ with
     symmetric InfoNCE. AdamW + cosine LR + gradient clipping.
  5. Evaluate R@1 / R@5 / R@10 retrieval after each epoch.

Dataset:
  Get the Kaggle mirror: https://www.kaggle.com/datasets/adityajn105/flickr8k
  Extract so DATA_ROOT contains:
      Images/*.jpg
      captions.txt   (CSV with header "image,caption")
  Then set DATA_ROOT below.
"""
import random
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader, Subset

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
)


def main():
    # ---------------- config ----------------
    DATA_ROOT = "./flickr8k"          # set to your dataset directory
    IMAGE_SIZE = 96
    PATCH_SIZE = 8                    # → (96/8)² = 144 patches
    MAX_SEQ_LEN = 32
    D_VIT = 256
    D_TEXT = 256
    D_LATENT = 128
    NUM_HEADS = 4
    NUM_LAYERS = 4                    # deeper than sanity, still tiny
    D_FF = 1024
    BATCH_SIZE = 64
    EPOCHS = 20
    LR = 5e-4
    WEIGHT_DECAY = 0.1                # CLIP-style, high
    GRAD_CLIP = 1.0
    VAL_SPLIT = 0.2
    MIN_WORD_FREQ = 3
    NUM_WORKERS = 2                   # set to 0 if you hit Windows DataLoader issues
    SEED = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(SEED)
    random.seed(SEED)

    # ---------------- tokenizer ----------------
    print("Building tokenizer …")
    captions = Flickr8kDataset.all_captions(DATA_ROOT)
    tokenizer = SimpleWordTokenizer()
    tokenizer.build_vocab(captions, min_freq=MIN_WORD_FREQ)
    print(f"Captions: {len(captions):,} | Vocab size: {tokenizer.vocab_size:,}")

    # ---------------- datasets ----------------
    train_ds_full = Flickr8kDataset(
        DATA_ROOT, tokenizer, MAX_SEQ_LEN,
        image_transform=make_train_transform(IMAGE_SIZE),
    )
    val_ds_full = Flickr8kDataset(
        DATA_ROOT, tokenizer, MAX_SEQ_LEN,
        image_transform=make_eval_transform(IMAGE_SIZE),
    )

    # Split by image so train and val share no images.
    image_names = sorted({name for name, _ in train_ds_full.samples})
    rng = random.Random(SEED)
    rng.shuffle(image_names)
    n_val_images = int(len(image_names) * VAL_SPLIT)
    val_images = set(image_names[:n_val_images])

    train_indices = [i for i, (n, _) in enumerate(train_ds_full.samples) if n not in val_images]
    val_indices = [i for i, (n, _) in enumerate(val_ds_full.samples) if n in val_images]
    train_ds = Subset(train_ds_full, train_indices)
    val_ds = Subset(val_ds_full, val_indices)
    print(f"Train pairs: {len(train_ds):,} | Val pairs: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, drop_last=True,
    )
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

    params = (
        list(image_encoder.parameters())
        + list(text_encoder.parameters())
        + list(proj_img.parameters())
        + list(proj_text.parameters())
        + list(temperature.parameters())
    )
    n_params = sum(p.numel() for p in params)
    print(f"Trainable params: {n_params:,}")

    # AdamW: decoupled weight decay (proper L2 regularization on weights).
    # High weight decay (0.1) is CLIP's choice — strong regularization on
    # small/medium contrastive datasets.
    optimizer = optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    # Cosine LR decay over all steps. Warmup omitted for simplicity; on
    # bigger runs you'd warm up for the first ~5% of steps.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * EPOCHS
    )

    # ---------------- training loop ----------------
    for epoch in range(EPOCHS):
        image_encoder.train()
        text_encoder.train()
        proj_img.train()
        proj_text.train()

        running = {"loss": 0.0, "acc_i2t": 0.0, "acc_t2i": 0.0}
        for step, (images, tokens, eos_pos) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            eos_pos = eos_pos.to(device, non_blocking=True)

            img_feat = image_encoder(images)
            text_feat = text_encoder(tokens, eos_pos)
            z_img = proj_img(img_feat)
            z_text = proj_text(text_feat)
            loss, m = symmetric_infonce_loss(z_img, z_text, temperature())

            optimizer.zero_grad()
            loss.backward()
            # Grad clipping keeps the contrastive loss from producing huge
            # updates on hard batches (especially early when τ is still high).
            torch.nn.utils.clip_grad_norm_(params, max_norm=GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            temperature.clamp_()

            for k in running:
                running[k] += m[k]

            if step % 50 == 0:
                print(
                    f"epoch {epoch} step {step:>4}/{len(train_loader)} | "
                    f"loss {m['loss']:.4f} | acc_i2t {m['acc_i2t']:.3f} | "
                    f"τ {temperature.tau():.4f} | "
                    f"lr {scheduler.get_last_lr()[0]:.2e}"
                )

        n_steps = len(train_loader)
        print(
            f"==> epoch {epoch} done | "
            f"avg loss {running['loss']/n_steps:.4f} | "
            f"avg acc_i2t {running['acc_i2t']/n_steps:.3f} | "
            f"τ {temperature.tau():.4f}"
        )

        metrics = compute_retrieval_metrics(
            image_encoder, text_encoder, proj_img, proj_text, val_loader, device
        )
        print("    val: " + " | ".join(f"{k} {v:.3f}" for k, v in metrics.items()))

    # ---------------- save ----------------
    Path("./checkpoints").mkdir(exist_ok=True)
    torch.save({
        "image_encoder": image_encoder.state_dict(),
        "text_encoder": text_encoder.state_dict(),
        "proj_img": proj_img.state_dict(),
        "proj_text": proj_text.state_dict(),
        "logit_scale": temperature.logit_scale.detach().cpu(),
        "tokenizer_vocab": tokenizer.id_to_word,
        "config": {
            "IMAGE_SIZE": IMAGE_SIZE, "PATCH_SIZE": PATCH_SIZE,
            "MAX_SEQ_LEN": MAX_SEQ_LEN, "D_VIT": D_VIT, "D_TEXT": D_TEXT,
            "D_LATENT": D_LATENT, "NUM_HEADS": NUM_HEADS,
            "NUM_LAYERS": NUM_LAYERS, "D_FF": D_FF,
        },
    }, "./checkpoints/mini_clip_flickr8k.pt")
    print("Saved checkpoint to ./checkpoints/mini_clip_flickr8k.pt")


if __name__ == "__main__":
    main()
