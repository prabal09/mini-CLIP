"""Step 3 sanity training — full mini-CLIP with both real encoders.

Both encoders are now real:
  • Image side: VisionTransformer (patch embed + CLS + pos embed + blocks + CLS pool).
  • Text side : TextTransformer from step 2.

Each concept has two RGB colors (top/bottom halves) and a unique topic
token. The model must learn:
  • The ViT extracts per-patch color, and CLS integrates both halves.
  • The text encoder maps the topic token to a discriminative embedding.
  • The contrastive loss aligns image and text embeddings concept-by-concept.

Expected behavior:
  • Loss decreases.
  • acc → 1.0 (may take a few hundred more steps than step 2 — more
    parameters, and the ViT must learn spatial integration).
  • τ shrinks.
  • Final loss should land near zero on this synthetic task.
"""
import torch
from torch import optim

from miniclip import (
    ProjectionHead,
    LearnedTemperature,
    symmetric_infonce_loss,
    TextTransformer,
    VisionTransformer,
    SyntheticImageTokens,
)


def main():
    torch.manual_seed(0)

    IMAGE_SIZE = 32
    PATCH_SIZE = 4         # → 64 patches
    D_VIT = 256
    D_TEXT = 256
    D_LATENT = 128
    BATCH_SIZE = 64
    STEPS = 2000
    LR = 1e-3
    MAX_SEQ_LEN = 16

    data = SyntheticImageTokens(
        num_concepts=256,
        image_size=IMAGE_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        pattern_len=4,
    )

    image_encoder = VisionTransformer(
        image_size=IMAGE_SIZE, patch_size=PATCH_SIZE,
        d_model=D_VIT, num_heads=4, num_layers=2, d_ff=512,
    )
    text_encoder = TextTransformer(
        vocab_size=data.vocab_size, max_seq_len=MAX_SEQ_LEN,
        d_model=D_TEXT, num_heads=4, num_layers=2, d_ff=512,
    )
    proj_img = ProjectionHead(D_VIT, D_LATENT)
    proj_text = ProjectionHead(D_TEXT, D_LATENT)
    temperature = LearnedTemperature(init_tau=0.07)

    params = (
        list(image_encoder.parameters())
        + list(text_encoder.parameters())
        + list(proj_img.parameters())
        + list(proj_text.parameters())
        + list(temperature.parameters())
    )
    optimizer = optim.Adam(params, lr=LR)

    n_params = sum(p.numel() for p in params)
    n_vit = sum(p.numel() for p in image_encoder.parameters())
    n_text = sum(p.numel() for p in text_encoder.parameters())
    print(f"Trainable params: {n_params:,}  (ViT {n_vit:,} + Text {n_text:,} + heads + τ)")
    print(f"{'step':>5} | {'loss':>7} | {'acc_i2t':>7} | {'acc_t2i':>7} | {'tau':>7} | {'log_scale':>9}")
    print("-" * 62)

    for step in range(STEPS):
        images, tokens, eos_pos = data.sample_batch(BATCH_SIZE)

        img_features = image_encoder(images)
        text_features = text_encoder(tokens, eos_pos)
        z_img = proj_img(img_features)
        z_text = proj_text(text_features)
        loss, m = symmetric_infonce_loss(z_img, z_text, temperature())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        temperature.clamp_()

        if step % 100 == 0 or step == STEPS - 1:
            print(
                f"{step:>5} | {m['loss']:>7.4f} | {m['acc_i2t']:>7.3f} | "
                f"{m['acc_t2i']:>7.3f} | {temperature.tau():>7.4f} | "
                f"{temperature.logit_scale.item():>9.4f}"
            )


if __name__ == "__main__":
    main()
