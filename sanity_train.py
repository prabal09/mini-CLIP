"""Step 1 sanity training — verify contrastive machinery on synthetic data.

Expected behavior:
  • Loss decreases monotonically (in expectation).
  • Top-1 in-batch retrieval accuracy → 1.0.
  • τ shrinks (logit_scale rises) as embeddings become discriminative —
    the model "earns" the right to sharpen the loss.

If any of these fail, the bug is in projection / temperature / loss —
not in the encoders we'll build later.
"""
import torch
from torch import optim

from miniclip import (
    ProjectionHead,
    LearnedTemperature,
    symmetric_infonce_loss,
    SyntheticPairs,
)


def main():
    torch.manual_seed(0)

    D_IMG, D_TEXT, D_LATENT = 512, 384, 128
    BATCH_SIZE = 64
    STEPS = 1500
    LR = 1e-3

    data = SyntheticPairs(num_concepts=256, d_img=D_IMG, d_text=D_TEXT)
    proj_img = ProjectionHead(D_IMG, D_LATENT)
    proj_text = ProjectionHead(D_TEXT, D_LATENT)
    temperature = LearnedTemperature(init_tau=0.07)

    params = (
        list(proj_img.parameters())
        + list(proj_text.parameters())
        + list(temperature.parameters())
    )
    optimizer = optim.Adam(params, lr=LR)

    print(f"{'step':>5} | {'loss':>7} | {'acc_i2t':>7} | {'acc_t2i':>7} | {'tau':>7} | {'log_scale':>9}")
    print("-" * 62)
    for step in range(STEPS):
        x_img, x_text = data.sample_batch(BATCH_SIZE)

        z_img = proj_img(x_img)
        z_text = proj_text(x_text)
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
