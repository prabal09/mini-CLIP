# mini-CLIP — Interview Q&A

Prep doc for the resume line:

> **Matching Engine with miniCLIP** — Reproduced a CLIP-style self-supervised contrastive
> foundation model from scratch in PyTorch on Flickr8k — symmetric InfoNCE loss with learnable
> temperature — as a paper-reproduction exercise for representation learning. Optimized the
> training loop with mixed-precision (AMP) and vectorized similarity-matrix ops, cutting
> per-epoch time.

Concrete config this project actually used (cite these — specifics kill pushback):
image size 96, patch 8 → 144 patches; text max_seq_len 32; d_model 256 (both towers),
shared latent dim 128; 4 heads, 4 layers, d_ff 1024; batch 64; 20 epochs; AdamW lr 5e-4,
weight decay 0.1, grad-clip 1.0, cosine LR (no warmup); learned temperature init τ=0.07 clamped
to τ ≥ 0.01; word-level tokenizer, min word freq 3; 80/20 split **by image**.

---

## 0. Traps in the resume phrasing (rehearse these first)

**Q: You wrote "self-supervised." Is image–text contrastive learning self-supervised?**
Strictly it's better called **natural-language-supervised** or **weakly supervised**. The caption is a
real signal paired with the image, not a pretext task derived from the input alone (as in SimCLR or
masked prediction). What makes it *feel* self-supervised is that no human hand-labeled categories —
supervision comes for free from naturally-occurring image–caption pairs. I'd describe it precisely as
"contrastive learning with natural-language supervision." I used "self-supervised" loosely on the
resume; the mechanism is contrastive alignment of two modalities.

**Q: Would you really call an 8k-image model a "foundation model"?**
No — by scale it isn't one. Real CLIP trained on ~400M pairs; Flickr8k is ~8k images / ~40k captions.
What I reproduced is the **method and architecture** of a foundation model (dual-encoder contrastive
pretraining that yields transferable, promptable embeddings), at a learning scale I could train and
inspect end-to-end. The value was understanding *why* the recipe works, not producing a deployable FM.

**Q: "Vectorized similarity-matrix ops" — what was un-vectorized before, and how much did it save?**
See the dedicated note at the end of this doc. Short version: the honest, defensible framing is that
the similarity matrix is a single batched matmul `z_img @ z_text.T` and both loss directions reuse it
(the second is just the transpose), and I use in-batch negatives so N examples produce N² comparisons
from one matmul with no Python loop over pairs. The real per-epoch speedup came from **AMP**, not from
"vectorization." If asked for a number, give the measured before/after seconds-per-epoch and the GPU;
if I don't have that number I should soften the claim rather than defend a figure I can't produce.

---

## 1. Contrastive learning / InfoNCE

**Q: Walk me through symmetric InfoNCE. Why two cross-entropy terms?**
Given a batch of N matched pairs, I L2-normalize both towers' embeddings and form the scaled similarity
matrix `logits = (z_img @ z_text.T) * exp(t)`, shape (N, N). Entry (i, j) is the cosine similarity of
image i with caption j, scaled by the inverse temperature. The correct match for row i is column i — the
diagonal — so the targets are just `arange(N)`. I take cross-entropy **twice**: once over each row
(image → text: "which caption matches this image?") and once over each column (text → image: "which
image matches this caption?"), then average. Two terms because the problem is symmetric — retrieval
should work in both directions — and a single direction would only constrain one of them. Off-diagonal
entries are the negatives.

**Q: What are the negatives?**
In-batch negatives: for image i, every other caption j≠i in the same batch is a negative, and vice
versa. No negative sampling or memory bank — the batch itself supplies N−1 negatives per anchor. That's
why batch size is effectively the number of negatives.

**Q: Why does batch size matter so much? What breaks at batch 8?**
The number of negatives ≈ batch size, and the contrastive signal sharpens with more negatives (the
softmax has to discriminate the positive against a harder, larger field). With batch 8 you get only 7
negatives per anchor; the task is too easy, embeddings underconstrain, and retrieval generalizes poorly.
This is exactly why CLIP used batch 32k and why I added the DDP all-gather path — to grow the effective
negative pool to global_batch = per_rank_batch × world_size without needing one giant GPU.

**Q: Where's the numerical-stability trick, and why does it matter late in training?**
`F.cross_entropy` applies log-softmax internally using the log-sum-exp identity
`log Σ exp(x_j) = M + log Σ exp(x_j − M)` with `M = max_j x_j`. Subtracting the row max before exp keeps
intermediates bounded. It matters late in training because τ shrinks toward its floor (0.01), so
inverse-temperature `exp(t)` grows toward 100 and logits span roughly [−100, 100]; `exp(100) ≈ 2.7e43`
would overflow without the max-subtraction. This is also the reason I keep the loss in FP32 even under
AMP (FP16 tops out near 65504).

**Q: Connection to mutual information / NCE?**
InfoNCE is a lower bound on the mutual information between the two views (image and caption): minimizing
it maximizes a bound on I(image; text). It's the noise-contrastive-estimation objective — classify the
true pair against noise (the in-batch negatives) — with the temperature-scaled cosine similarity as the
critic. I wouldn't overclaim tightness of the bound; the practical point is it pulls matched pairs
together and pushes mismatched pairs apart on the unit sphere.

**Q: How would you add hard-negative mining, and would it help here?**
You'd bias the negative set toward high-similarity mismatches (e.g., mine within-batch hardest columns,
or maintain a queue à la MoCo). On Flickr8k the bigger lever is simply more in-batch negatives (larger
batch / all-gather), so I'd reach for that first. Hard-negative mining also risks sampling **false**
negatives here, because Flickr8k has ~5 captions per image — two captions of the same image look like a
"hard negative" but are actually positives. So I'd be cautious: fix the false-negative issue before
mining harder ones.

---

## 2. Temperature

**Q: Why make temperature learnable? What if it's fixed wrong?**
Temperature τ controls how peaked the softmax over similarities is. Too high (soft) and even the correct
pair gets little relative weight — gradients are weak and training stalls. Too low (sharp) and the loss
is dominated by the single hardest negative, training gets unstable and can collapse. The right τ shifts
during training, so I learn it jointly rather than guessing a schedule. This is exactly what CLIP does.

**Q: You store log(1/τ) (the logit scale), not τ. Why log space?**
Two reasons. (1) Positivity for free: τ = exp(−t) is positive for any real t, so the optimizer can range
over all of ℝ without constraints. (2) Multiplicative updates: a gradient step on t moves τ by a
*factor*, which behaves well across orders of magnitude (τ = 0.5 vs 0.01) — additive steps on τ directly
would be ill-conditioned near zero. Forward returns `exp(t)`, which multiplies the cosine similarities.

**Q: Why clamp the logit scale at log(100), i.e. τ ≥ 0.01?**
To prevent τ → 0 collapse. As τ shrinks, logits blow up and gradients explode; left unchecked the model
drives τ down to overfit the batch. CLIP caps the logit scale at 100 (τ ≥ 0.01). I clamp **in place after
each optimizer step**, so the parameter itself never drifts past the ceiling rather than just clamping
the forward value.

**Q: What did τ actually do over training?**
It starts at 0.07 (t ≈ 2.66) and generally decreases as the model gets confident about matched pairs —
the model wants sharper softmaxes once alignment is good. I log τ every 50 steps and at epoch end
precisely so I can point to this curve. [Fill in your observed final τ from the run — e.g. "settled
around 0.02–0.03 and hit the 0.01 floor only if it collapsed."] If it slammed into the floor early,
that's a signal of too-small batch / too-few negatives.

---

## 3. Text encoder

**Q: Why take the EOS token's hidden state as the sentence embedding?**
It's CLIP's convention and it's principled given the causal mask. With causal self-attention, position i
can only attend to positions ≤ i, so **only the last real token (EOS) has attended to the entire
sequence**. Its final hidden state is therefore the one position that has seen the whole caption — the
natural summary vector. I pass explicit `eos_positions` per row and gather that slot, because captions
are padded to max_seq_len and EOS sits at different indices per example.

**Q: Isn't bidirectional (BERT-style) better for a pure embedding model? Why causal?**
For representation quality alone, bidirectional pooling (or mean-pooling) is a defensible, often better
choice — every token sees full context. CLIP deliberately used a causal GPT-style text encoder, partly
for architectural consistency with autoregressive LMs and to keep the option of generative pretraining
open. I reproduced CLIP's choice on purpose. If I were optimizing this encoder in isolation for retrieval,
trying a bidirectional encoder with mean/attention pooling would be a reasonable ablation.

**Q: Why add token and positional embeddings instead of concatenating?**
Concatenation would double d_model and force a projection to fold it back — more parameters, no clear
benefit. Addition lets the network allocate feature dimensions to disentangle "what token" from "where"
as needed, and it's the design that's held up across Transformer variants for years. Both are learned
`nn.Embedding` tables here.

**Q: Pre-LN or post-LN blocks, and why?**
Pre-LN: `x = x + Attn(LN(x)); x = x + FFN(LN(x))`. The residual stream stays unnormalized and each
sublayer reads a normalized view. Pre-LN is more stable at depth and is what GPT-2 and modern CLIP
variants use; post-LN needs careful warmup to avoid early instability. I also do a final LayerNorm before
pooling.

**Q: Attention details?**
Standard scaled dot-product multi-head attention with a single fused QKV projection (`Linear(d, 3d)` then
chunk) — one matmul instead of three. Scores are `QKᵀ/√d_head` so the variance stays ≈1 regardless of
head size; softmax; then `·V`; heads run on disjoint d_head subspaces and are concatenated and projected
by W_O. Causal mask is an upper-triangular −∞ fill applied before softmax (text tower only).

---

## 4. Image encoder (ViT)

**Q: Walk through the ViT forward pass.**
Patchify a 96×96 image into 8×8 patches → 144 patches, each linearly projected to d_model=256 (I use a
Conv2d with kernel=stride=patch_size, which is mathematically identical to extracting non-overlapping
patches and multiplying by one matrix, but is a single op — the original ViT trick). Prepend a learnable
CLS token → sequence length 145. Add a learned positional embedding table. Run K non-causal Transformer
blocks (every patch attends to every patch). Final LayerNorm. Pool the CLS slot (position 0) as the image
embedding.

**Q: Why a CLS token instead of mean-pooling the patches?**
With full (non-causal) attention, the CLS token has no intrinsic spatial content — it's a dedicated
readout slot the model can fill with whatever global summary minimizes the contrastive loss, aggregating
across all patches through attention. Mean-pooling is a valid alternative and sometimes competitive; CLS
gives the model a learnable, flexible pooling function rather than a fixed average. It's the ViT/CLIP
convention, so I matched it.

**Q: Why learned positional embeddings, not sinusoidal?**
Patchify + linear projection discards spatial arrangement; positional embeddings restore "where each
patch was." Learned tables are simpler and are what ViT/CLIP use. Sinusoidal would generalize better to
unseen image sizes, but I train at a fixed 96×96 so I don't need that; it wasn't worth the added
complexity here.

**Q: Why is the Conv2d equivalent to patch embedding?**
A Conv2d with kernel_size = stride = P slides over non-overlapping P×P tiles and applies the same
(P²·C → d_model) linear map to each — which is exactly "cut into patches, flatten, project." One
operation, no explicit reshape-and-matmul.

---

## 5. Projection heads & the shared space

**Q: What do the projection heads do and why?**
Each tower outputs 256-dim features in its own space; the heads are `Linear(256 → 128, bias=False)`
mapping both into a **shared** 128-dim latent, followed by L2-normalization. You need a common space to
compare across modalities, and a separate learned projection lets each tower specialize its backbone
while the head handles alignment.

**Q: Why L2-normalize before the similarity matrix?**
After L2-norm, the dot product equals cosine similarity, bounded in [−1, 1], and all embeddings live on
the unit hypersphere. This decouples "direction" (what we compare) from "magnitude," and it's what makes
the temperature the single knob controlling logit scale. Without normalization, dot products would be
unbounded, magnitudes would silently act as a per-example temperature, and the softmax would be
ill-behaved.

**Q: Why no bias in the projection?**
A bias shifts every embedding off the sphere by a constant, and the following L2-norm would largely undo
it — so it's wasted parameters. Omitting it is cleaner and standard.

---

## 6. Training loop & AMP

**Q: Explain AMP. What's in FP16 vs FP32, and what's the GradScaler for?**
AMP runs the heavy matmuls (the encoder forward passes) under `autocast` in FP16 to use tensor cores,
while keeping numerically sensitive ops in FP32. In my loop the encoders run inside `autocast`, then I
**cast features back to `.float()`** before the projection heads, L2-norm, similarity matrix, and loss —
all FP32 — because those involve the large-logit softmax that would overflow in FP16. The **GradScaler**
handles the other FP16 hazard: small gradients underflowing to zero. It multiplies the loss by a large
scale before backward (so gradients land in FP16's representable range), then I unscale before the
optimizer step; it also auto-tunes the scale and **skips the step if it detects inf/nan** grads.

**Q: Order of unscale vs grad-clip — why does it matter?**
I unscale the gradients **before** `clip_grad_norm_`. If I clipped first, the threshold (max_norm=1.0)
would be applied to loss-scaled gradients — i.e., a meaningless, scale-dependent magnitude. Unscaling
first means the clip acts on true-magnitude gradients.

**Q: FP16 vs BF16 — why did you default to FP16, and when would you switch?**
FP16 has a narrow dynamic range and needs the GradScaler; BF16 has the same exponent range as FP32, so
it usually needs no scaler and is more robust — but it needs Ampere+ hardware and has fewer mantissa
bits. My code exposes `AMP_DTYPE` and I default to FP16 for broad GPU compatibility; on an Ampere+ card
I'd switch to BF16 and could drop the scaler. AMP is also gated to CUDA only — on CPU, FP16 typically
runs slower (no tensor cores, conversion overhead dominates), so the context manager becomes a no-op.

**Q: Did AMP change results or just speed? How did you confirm?**
Goal is speed/memory at (near-)equal quality. Because I keep the loss and normalization in FP32 and use
the scaler, accuracy and R@K should track the FP32 run within noise. I'd confirm by comparing final
val R@1/R@5 and the loss/accuracy curves between an AMP and a non-AMP run with the same seed. [Fill in:
your measured before/after seconds-per-epoch + GPU, and that R@K matched within noise.]

**Q: You mention gradient accumulation — why, and the contrastive-loss gotcha?**
Accumulation lets you simulate a larger *optimizer* batch on limited memory by summing gradients over
several micro-batches before stepping. The gotcha specific to contrastive learning: it does **not**
enlarge the negative set. In-batch negatives only come from examples that share a forward pass, so
accumulating 4 micro-batches of 64 gives you 4 gradient contributions but still only 63 negatives per
anchor — not 255. To actually grow negatives you need them in the *same* similarity matrix, which is why
the real fix is the DDP all-gather, not accumulation.

**Q: Optimizer, schedule, regularization — and why?**
AdamW (decoupled weight decay = proper L2 on weights, not folded into the Adam moment estimates), lr
5e-4, **weight decay 0.1** — deliberately high, CLIP's choice, strong regularization for small/medium
contrastive data. Cosine LR decay over all steps, grad-clip at norm 1.0 for stability. I **omitted
warmup** for simplicity; on a larger run I'd warm up the first ~5% of steps, since Adam's early
second-moment estimates are noisy and pre-LN + warmup is the standard stable recipe.

---

## 7. Evaluation / retrieval

**Q: How do you measure success? Define R@K here.**
Recall@K on the val set, both directions. I encode every val (image, caption) pair, build the full N×N
cosine-similarity matrix, and for each image check whether its true caption is in the top-K columns
(image→text R@K), and symmetrically for each caption (text→image). I report R@1/R@5/R@10. (τ is
irrelevant for ranking — softmax is monotonic in the similarities, so argmax/top-K don't depend on it.)

**Q: What's the caveat in your retrieval metric?**
I use the **exact-pair** definition: caption at index i is the one correct answer for image i. But
Flickr8k has ~5 captions per image, so a stricter, more standard metric would treat *any* of an image's 5
captions as correct (group-by-image). My exact-pair version slightly *understates* true retrieval quality
(the model can retrieve a different valid caption of the same image and be marked wrong). I chose it for
simplicity/readability at this scale and I'm explicit that the grouped metric is the "correct" one for
reporting against the literature.

**Q: What's the difference between zero-shot retrieval and zero-shot classification?**
Retrieval ranks a held-out set of candidate captions/images by similarity. Zero-shot *classification* is
the famous CLIP trick: turn each class label into a text prompt ("a photo of a dog"), encode the prompts,
and classify an image by nearest prompt embedding — no classifier head, no fine-tuning. Both use the same
frozen embeddings; classification is retrieval against a set of synthetic label-prompts.

**Q: How did you split train/val, and is there leakage?**
I split **by image**, not by pair: 20% of *images* go to val, and all captions of a val image go to val.
If I'd split by pair, the same image could appear in both train and val (via different captions), leaking
visual content. Splitting by image prevents that. (I use two Dataset instances — train transform vs.
deterministic eval transform — indexed by disjoint image sets.)

---

## 8. Data / tokenizer / augmentation

**Q: Word-level tokenizer — what do you lose vs. BPE, and how is OOV handled?**
I built a word-level tokenizer (special tokens PAD/BOS/EOS/UNK at ids 0–3, real words by descending
frequency, min freq 3). It's pedagogically clear — every token is a readable word — but it has a fixed
closed vocabulary: rare or unseen words map to UNK, and it can't compose subwords. BPE would handle OOV
gracefully, shrink the vocab, and share morphology, at the cost of readability. For Flickr8k's small,
repetitive vocabulary the tradeoff is acceptable; BPE would be the upgrade for anything open-domain.

**Q: Caption layout fed to the encoder?**
`[BOS, w1, …, wk, EOS, PAD…]` padded to max_seq_len=32, captions truncated to fit BOS+EOS. I return the
EOS position per example so the text encoder can pool the right slot.

**Q: What augmentations, and why is text augmentation harder?**
Image side: RandomResizedCrop (scale 0.6–1.0) to force scale/framing invariance, horizontal flip (free 2×
on natural photos), light ColorJitter to break color-statistic shortcuts, then ImageNet normalization.
Eval is deterministic resize + center-crop. I deliberately did **not** augment text — small word-level
edits (synonym swap, deletion) risk changing the caption's meaning and breaking the true pairing, whereas
a cropped/flipped image is still the same image. Text augmentation is semantically fragile, so I left it
out.

**Q: 5 captions per image — how do you form positives, and what's the risk?**
Each (image, caption) row is an independent training pair, so an image appears in ~5 pairs. The risk: two
captions of the same image can land in the same batch, where they become each other's "negatives" even
though both correctly describe that image — **false negatives** that the loss wrongly pushes apart. At
batch 64 over ~8k images this is rare but real; mitigations would be grouping by image in the sampler or a
loss that tolerates multiple positives. Worth naming proactively — it shows I understand the data.

---

## 9. Distributed training (DDP) — off-resume but in the repo

**Q: What's the one subtlety in DDP + contrastive loss beyond wrapping in DDP?**
Naive DDP undertrains InfoNCE: DDP parallelizes the *batch*, but each rank computes the loss only against
its **local** negatives, so the effective negative pool stays at per-rank batch no matter how many GPUs —
you get throughput scaling but not the contrastive benefit of a bigger batch. Fix: **all-gather the
projected embeddings across ranks before the loss**, so every rank builds a (global_batch, global_batch)
similarity matrix. Global batch = per-rank × world_size.

**Q: But all_gather detaches gradients — how do you keep them flowing?**
`dist.all_gather` returns detached tensors, so gradients to non-local shards are lost. The OpenCLIP trick:
after gathering, overwrite the local slot of the gathered list with the *local* tensor (which still has
its grad_fn). Then the loss's gradient w.r.t. the local shard flows back into the local encoder; every
rank does this symmetrically, and DDP's standard parameter all-reduce averages gradients across ranks.

**Q: Other DDP correctness details you handled?**
`DistributedSampler.set_epoch(epoch)` each epoch (otherwise every epoch shuffles identically — a classic
silent bug); same seed on all ranks so vocab and train/val split are identical; validation on rank 0 only
with a `dist.barrier()` so other ranks don't race into the next epoch's all_gather; NCCL backend on
CUDA/Linux, gloo fallback on Windows/CPU; temperature clamp stays in sync across ranks for free because
DDP synchronizes the gradients that produced it.

---

## 10. Big-picture / critique

**Q: CLIP's whole point was scale. What's reproducible at 8k and what isn't?**
Reproducible: the mechanics — that symmetric InfoNCE aligns two modalities, that learned temperature
self-regulates, that more negatives help, that the loss converges and retrieval works above chance.
Not reproducible: the emergent zero-shot generalization that made CLIP famous — that needs the data
diversity of hundreds of millions of pairs. At 8k the model memorizes a narrow distribution; it won't
transfer to arbitrary concepts. The project was about *understanding the recipe*, not matching the
capability.

**Q: If R@1 is low, how do you diagnose?**
Walk the stack: (1) is training loss actually decreasing and in-batch top-1 accuracy climbing toward 1?
If not, it's optimization (lr, warmup, τ collapse). (2) If train aligns but val doesn't, it's
generalization — too little data / too much capacity → more augmentation, weight decay, smaller model.
(3) Check τ didn't hit the floor early (too few negatives → bump batch or use the all-gather path).
(4) Sanity-check the data pipeline (right EOS position, no train/val leakage, captions matched to images).
I built the synthetic sanity layer first precisely so I could isolate loss/temperature bugs from
encoder/data bugs.

**Q: Why build a synthetic sanity layer before real encoders?**
To prove the *objective* in isolation. I trained the projection heads + InfoNCE + learned temperature on
synthetic image/caption vectors with known structure, so I could confirm the loss converges, in-batch
accuracy → 1, and τ behaves — before introducing any encoder or data-loading complexity. When something
broke later, I knew it wasn't the loss. This staged bring-up (sanity → text encoder → ViT → real data →
AMP → DDP) is the main engineering story of the project.

**Q: What would you change to make this production-useful?**
Bigger and more diverse data (the real lever); BPE tokenizer; pretrained or larger backbones; the
group-by-image retrieval metric and false-negative-aware loss; BF16 + the all-gather DDP path for a large
effective batch; warmup; and an ANN index (FAISS) for retrieval at scale instead of a dense N×N matrix.

---

## Appendix — the "vectorized similarity-matrix ops" phrase (read the main note in chat)

Defensible framing if pressed: one batched matmul `z_img @ z_text.T` produces the full N×N similarity
matrix; both loss directions reuse it (`logits_per_text` is just the transpose); in-batch negatives mean
N examples yield N² comparisons with no Python loop. The measurable per-epoch win, though, is AMP — be
honest about that split, and bring a real before/after number.
