import torch


@torch.no_grad()
def compute_retrieval_metrics(
    image_encoder,
    text_encoder,
    proj_img,
    proj_text,
    loader,
    device,
):
    """Compute R@K image↔text retrieval metrics on the val set.

    Procedure:
      1. Encode every (image, caption) pair in the loader → (z_img, z_text)
         both of shape (N, d_latent).
      2. Compute the full N×N similarity matrix S = z_img · z_textᵀ.
         (τ doesn't matter for argmax — softmax is monotonic in S.)
      3. For each row i (image), check whether column i is in top-K
         → R@K for image→text.
      4. For each column j (caption), check whether row j is in top-K
         → R@K for text→image.

    Note: this uses the *exact-pair* definition of correctness — caption
    at index i is THE positive for image at index i. Since Flickr8k has
    ~5 captions per image, a stricter metric would group by image. For
    a learning-scale project, the exact-pair version is informative and
    easier to read.
    """
    image_encoder.eval()
    text_encoder.eval()
    proj_img.eval()
    proj_text.eval()

    all_z_img = []
    all_z_text = []
    for images, tokens, eos_pos in loader:
        images = images.to(device)
        tokens = tokens.to(device)
        eos_pos = eos_pos.to(device)

        img_feat = image_encoder(images)
        text_feat = text_encoder(tokens, eos_pos)
        all_z_img.append(proj_img(img_feat).cpu())
        all_z_text.append(proj_text(text_feat).cpu())

    z_img = torch.cat(all_z_img, dim=0)
    z_text = torch.cat(all_z_text, dim=0)
    sims = z_img @ z_text.t()
    N = sims.shape[0]
    targets = torch.arange(N).unsqueeze(1)

    metrics: dict[str, float] = {}
    for k in (1, 5, 10):
        kk = min(k, N)
        topk_i2t = sims.topk(kk, dim=1).indices
        topk_t2i = sims.t().topk(kk, dim=1).indices
        metrics[f"R@{k}_i2t"] = (topk_i2t == targets).any(dim=1).float().mean().item()
        metrics[f"R@{k}_t2i"] = (topk_t2i == targets).any(dim=1).float().mean().item()
    return metrics
