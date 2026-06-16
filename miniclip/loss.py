import torch
import torch.nn.functional as F


def symmetric_infonce_loss(
    z_img: torch.Tensor,
    z_text: torch.Tensor,
    logit_scale: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Symmetric InfoNCE contrastive loss used by CLIP.

    Inputs are L2-normalized embeddings of shape (N, d). The similarity
    matrix S has shape (N, N):

        Sᵢⱼ = ⟨z_imgᵢ, z_textⱼ⟩ · exp(t)

    The matched pair is the diagonal (i, i). Off-diagonal entries are
    in-batch negatives. We compute two cross-entropies:

        image → text:  for each image i, target text is j = i
        text → image:  for each text  j, target image is i = j

    and average them. F.cross_entropy applies log-softmax internally,
    which uses the log-sum-exp trick:

        log Σⱼ exp(xⱼ) = M + log Σⱼ exp(xⱼ − M),   M = max_j xⱼ

    This subtraction prevents overflow when logits become large
    (e.g., late in training when τ is small).

    Returns (loss, metrics_dict). The dict includes in-batch top-1
    accuracy as a diagnostic — should approach 1.0 if the model is
    learning to align matched pairs.
    """
    N = z_img.shape[0]

    # (N, N) — row i, col j is the scaled similarity of image i with text j.
    logits_per_image = (z_img @ z_text.t()) * logit_scale
    logits_per_text = logits_per_image.t()

    targets = torch.arange(N, device=z_img.device)

    loss_i2t = F.cross_entropy(logits_per_image, targets)
    loss_t2i = F.cross_entropy(logits_per_text, targets)
    loss = (loss_i2t + loss_t2i) / 2.0

    with torch.no_grad():
        acc_i2t = (logits_per_image.argmax(dim=1) == targets).float().mean().item()
        acc_t2i = (logits_per_text.argmax(dim=1) == targets).float().mean().item()

    return loss, {
        "loss": loss.item(),
        "loss_i2t": loss_i2t.item(),
        "loss_t2i": loss_t2i.item(),
        "acc_i2t": acc_i2t,
        "acc_t2i": acc_t2i,
    }
