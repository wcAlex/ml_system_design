# 06 — Model Development

## Two Core Training Approaches

This section covers how to actually train your embedding model. Two approaches dominate production systems:

1. **Contrastive / Metric Learning** (image-to-image similarity)
2. **CLIP Fine-tuning** (multi-modal, image + text)

Both can be used together — CLIP gives you the base, contrastive fine-tuning adapts it to your product domain.

---

## Approach 1: Contrastive Metric Learning

### Goal
Learn a function `f(image) → embedding` such that:
- `distance(f(img_A), f(img_B))` is small if A and B are visually similar
- `distance(f(img_A), f(img_C))` is large if A and C are different

### Loss Functions

#### Option A: Triplet Loss (Classic)
```
Input: (Anchor, Positive, Negative) triplet
Loss = max(0, d(a,p) - d(a,n) + margin)

where:
  a = anchor embedding (query image)
  p = positive embedding (visually similar item)
  n = negative embedding (different item)
  margin = 0.2 (hyperparameter)
```

**Intuition:** Pull anchor closer to positive, push it away from negative.

**Training example:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # All inputs: (batch_size, embedding_dim), L2-normalized
        pos_dist = 1 - F.cosine_similarity(anchor, positive)  # 1 - cos_sim
        neg_dist = 1 - F.cosine_similarity(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
```

**Pros:** Simple, widely understood
**Cons:** Triplet mining is tricky — random triplets are too easy (model learns nothing)

#### Option B: InfoNCE / Contrastive Loss (Modern, Preferred)
```
For a batch of N (anchor, positive) pairs:
  - 2N total embeddings
  - Positives: (anchor_i, positive_i) pairs
  - Negatives: all other 2N-2 items in the batch

Loss = -log [ exp(sim(a_i, p_i) / τ) / Σ_j exp(sim(a_i, e_j) / τ) ]
```

Where τ (temperature, ~0.07) controls the sharpness of the distribution.

**Intuition:** Like a softmax classification — treat finding the correct positive as a classification problem among all negatives in the batch.

```python
def info_nce_loss(embeddings_a, embeddings_b, temperature=0.07):
    """
    embeddings_a: (N, D) — anchor embeddings, L2-normalized
    embeddings_b: (N, D) — positive embeddings, L2-normalized
    """
    N = embeddings_a.size(0)
    # Similarity matrix: (N, N)
    logits = torch.matmul(embeddings_a, embeddings_b.T) / temperature
    # Labels: diagonal is the positive pair for each anchor
    labels = torch.arange(N, device=embeddings_a.device)
    # Symmetric loss
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    return (loss_a + loss_b) / 2
```

**Why better than triplet:** Uses all in-batch negatives efficiently. Larger batch = harder negatives. Used by SimCLR, MoCo, CLIP.

**Key insight:** This is exactly the loss CLIP uses. With a batch size of 4096, each image has 4095 negatives — very hard training signal.

### Training Setup for Metric Learning

```
Data:     Same-SKU pairs as positives; hard negatives from offline mining
Backbone: Frozen or partially frozen (fine-tune last 4 layers of ViT)
Head:     Linear projection → 512-d → L2 normalize
Optimizer: AdamW, lr=1e-4 (backbone) / 1e-3 (head)
Batch:    256–1024 (larger = harder negatives for InfoNCE)
Epochs:   10–30 for fine-tuning; more for from-scratch
Schedule: Cosine LR decay with warmup
```

---

## Approach 2: CLIP Fine-tuning

### Why Fine-tune CLIP?

Pretrained CLIP is strong but suffers from **domain shift** for product images:
- Web photos ≠ product photos (lighting, background, angle)
- Product attributes (exact colorways, material names) not well-represented in web captions

Fine-tuning adapts the model to your domain with minimal data.

### Option A: Full Fine-tuning
- Update all weights of both image and text encoders
- Requires large dataset (>1M pairs) to avoid catastrophic forgetting
- Best final performance when data is available

### Option B: Adapter / LoRA Fine-tuning (Recommended)
Insert small trainable layers into frozen backbone:

```python
class CLIPAdapter(nn.Module):
    """
    Adapter layers inserted into frozen CLIP image encoder.
    Only train the adapter (<<1% of total params).
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.fc2 = nn.Linear(dim // reduction, dim)
        self.alpha = nn.Parameter(torch.tensor(0.2))  # learnable blend

    def forward(self, x):
        residual = self.fc2(F.relu(self.fc1(x)))
        return x + self.alpha * residual  # residual connection
```

- **Pros:** Only train adapters — 10–100x fewer parameters. No catastrophic forgetting.
- **Cons:** Smaller capacity than full fine-tuning

### Option C: Linear Probing
- Freeze everything, train only a linear head on top of frozen CLIP
- Fastest to train, minimal compute
- Worse performance than full/adapter fine-tuning
- Useful as a strong baseline or for very small datasets

### Fine-tuning Data Sources
1. (product_image, product_title) pairs from your catalog — contrastive loss
2. (query_image, clicked_product_title) pairs from logs — user intent signal
3. Hard negative pairs: products with similar titles but different visual appearance

---

## Approach 3: Two-Tower Model (Multi-modal Production System)

For production systems that need to handle both image queries and text queries:

```
Query Side (online)              Catalog Side (offline, precomputed)
────────────────────             ──────────────────────────────────
Image → Image Encoder ──►        Product Image → Image Encoder ──►
                       │                                            ├── Combined product embedding
Text → Text Encoder ──►│         Product Text → Text Encoder ──►──►
       (optional)      │
                       │
              Fused query embedding
                       │
                       ▼
              ANN Search → Top-K candidates
```

**Fusion strategies for query embedding:**
- Simple average: `q_embed = (img_embed + text_embed) / 2`
- Weighted: `q_embed = α·img_embed + (1-α)·text_embed` (α learned or user-controlled)
- Cross-attention: Let image and text attend to each other before fusion (more powerful, higher latency)

---

## Training Curriculum (Recommended Order)

```
Stage 1: Pretrained CLIP ViT-B/16 (out of box)
         → Establish offline baseline (Recall@10)

Stage 2: Linear probing on product catalog
         → Fast adaptation, ~1 day training

Stage 3: Adapter fine-tuning with InfoNCE loss
         → Domain adaptation with (image, title) pairs

Stage 4: Hard negative mining pass
         → Mine hard negatives, re-train Stage 3

Stage 5: Behavior-signal fine-tuning
         → Incorporate click/purchase pairs for user intent
```

Each stage is an A/B test opportunity — validate before moving to next.

---

## Hyperparameter Summary

| Hyperparameter | Typical Range | Notes |
|---|---|---|
| Embedding dim | 256–768 | 512 is a good default |
| Temperature (τ) | 0.05–0.1 | Lower = harder, can collapse |
| Batch size | 256–4096 | Larger = harder negatives |
| Learning rate (backbone) | 1e-5 – 5e-5 | Fine-tune conservatively |
| Learning rate (head) | 1e-4 – 1e-3 | Head trains faster |
| Warmup steps | 500–2000 | Prevents early instability |
| Margin (triplet) | 0.1–0.5 | Tune on validation set |

---

## Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| Embeddings collapse (all close together) | Too low temperature or LR | Increase τ, reduce LR |
| Random-looking results | Not enough hard negatives | Add hard negative mining |
| Great offline metrics but bad online CTR | Distribution shift in user photos | Add user-photo augmentation |
| Category confusion (shoes returned for bags) | Weak category signal | Add category classification head |

---

## Interview Checkpoint

1. **"How do you prevent the embedding space from collapsing?"**
   - Use InfoNCE with sufficient batch size (256+), check embedding distribution via mean and variance, use L2 normalization, avoid too-low temperature.

2. **"Why not use cross-entropy classification loss for retrieval?"**
   - Classification trains a fixed set of classes; retrieval must generalize to unseen items. Metric learning/contrastive loss creates a generalizable embedding space. New products get reasonable embeddings without retraining.

3. **"How do you handle the cold-start for new categories?"**
   - CLIP generalizes zero-shot because it has seen these categories in web text. For niche categories (e.g., artisan goods), add a few labeled examples and use few-shot fine-tuning or prompting.

4. **"How do you know when to stop training?"**
   - Monitor Recall@10 on a held-out evaluation set. Use early stopping with patience=3 (stop if no improvement in 3 epochs). Track training/eval loss ratio to detect overfitting.
