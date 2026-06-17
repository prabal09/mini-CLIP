"""Step 2 sanity training — verify the text Transformer end-to-end.

Plugs the new TextTransformer into the contrastive setup from step 1.
Image side is still a fake feature (we'll build the ViT in step 3).
Text side now passes through token embedding, positional embedding,
causal self-attention blocks, final LN, and EOS pooling.

Expected behavior — identical signature to step 1:
  • Loss decreases.
  • Top-1 accuracy → 1.0.
  • τ shrinks as embeddings separate (slower than step 1: the text
    encoder has many more parameters to organize).
If all hold, the Transformer architecture is wired correctly.
"""
import torch
from torch import optim

from miniclip import (
    ProjectionHead,
    LearnedTemperature,
    symmetric_infonce_loss,
    SyntheticTextPairs,
    TextTransformer,
)


def main():
    torch.manual_seed(0)

    D_IMG = 512
    D_LATENT = 128
    D_TEXT = 256          # internal width of the text Transformer
    BATCH_SIZE = 64
    STEPS = 1500
    LR = 1e-3
    MAX_SEQ_LEN = 16

    data = SyntheticTextPairs(
        num_concepts=256, d_img=D_IMG, max_seq_len=MAX_SEQ_LEN, pattern_len=4
    )

    text_encoder = TextTransformer(
        vocab_size=data.vocab_size,
        max_seq_len=MAX_SEQ_LEN,
        d_model=D_TEXT,
        num_heads=4,
        num_layers=2,
        d_ff=512,
    )
    proj_img = ProjectionHead(D_IMG, D_LATENT)
    proj_text = ProjectionHead(D_TEXT, D_LATENT)
    temperature = LearnedTemperature(init_tau=0.07)

    params = (
        list(text_encoder.parameters())
        + list(proj_img.parameters())
        + list(proj_text.parameters())
        + list(temperature.parameters())
    )
    optimizer = optim.Adam(params, lr=LR)

    n_params = sum(p.numel() for p in params)
    print(f"Trainable params: {n_params:,}")
    print(f"{'step':>5} | {'loss':>7} | {'acc_i2t':>7} | {'acc_t2i':>7} | {'tau':>7} | {'log_scale':>9}")
    print("-" * 62)

    for step in range(STEPS):
        x_img, tokens, eos_pos = data.sample_batch(BATCH_SIZE)

        text_features = text_encoder(tokens, eos_pos)
        z_img = proj_img(x_img)
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
