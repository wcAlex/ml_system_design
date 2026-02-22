# CLIP & Contrastive Visual-Language Models

## 1. What Is CLIP?

**CLIP (Contrastive Language-Image Pretraining)** is a model from OpenAI (2021) that learns to align images and text in a shared embedding space by training on 400 million (image, caption) pairs scraped from the web.

The core insight: instead of teaching a model "this image is a cat," teach it that **the image of a cat and the text "a photo of a cat" should be close together** in vector space.

```
                    CLIP Embedding Space
                    ────────────────────────────────────────

  Image of a sneaker  ──► [0.23, -0.11, 0.87, ...]  ◄── "Nike Air Force 1 sneaker"
  Image of a handbag  ──► [0.71,  0.44, 0.02, ...]  ◄── "leather handbag"

  cosine_sim(sneaker_img_embed, sneaker_text_embed) → high (~0.95)
  cosine_sim(sneaker_img_embed, handbag_text_embed) → low (~0.12)
```

---

## 2. Architecture

CLIP has two independent encoders that are trained jointly:

```
                 ┌──────────────────────────────────────────────────┐
                 │                     CLIP                         │
                 │                                                  │
  Image ────────►│  Image Encoder (ViT or ResNet)                  │
                 │  - Input: 224×224 RGB image                      │──► 512-d image embed (L2-norm)
                 │  - Output: [CLS] token embedding                 │
                 │                                                  │
  Text ─────────►│  Text Encoder (Transformer)                     │
                 │  - Input: tokenized text (max 77 tokens)         │──► 512-d text embed (L2-norm)
                 │  - Output: [EOS] token embedding                 │
                 │                                                  │
                 │  Contrastive Loss (InfoNCE)                      │
                 │  - Pull matched pairs together                   │
                 │  - Push non-matched pairs apart                  │
                 └──────────────────────────────────────────────────┘
```

### Image Encoder Options

| Encoder | Notes |
|---|---|
| **ViT-B/32** | 16×16 patches, fast, lower quality |
| **ViT-B/16** | Smaller patches = finer detail, default choice |
| **ViT-L/14** | Large ViT, best quality, higher latency |
| **ResNet-50/101** | CNN alternative; less common in modern CLIP |

### Text Encoder

- Transformer with 12 layers, 512-d embedding
- BPE tokenizer, max 77 tokens
- The embedding at the `[EOS]` token position represents the full sentence

---

## 3. Training: InfoNCE Contrastive Loss

Given a batch of N (image, text) pairs:

```
Batch of N=4 pairs:
  (img_sneaker, "Nike sneaker")
  (img_dress,   "floral dress")
  (img_bag,     "leather bag")
  (img_watch,   "gold watch")

Similarity matrix (4×4): how similar is each image to each text?

                "Nike sneaker"  "floral dress"  "leather bag"  "gold watch"
  img_sneaker [     0.95            0.08            0.12           0.05    ]  ← row sum softmax
  img_dress   [     0.06            0.93            0.11           0.09    ]
  img_bag     [     0.10            0.07            0.91           0.13    ]
  img_watch   [     0.04            0.11            0.09           0.94    ]

Diagonal = correct pairs (should be high)
Off-diagonal = negatives (should be low)
```

**Loss:**
```
L = -1/N × Σᵢ [ log(softmax(sim(img_i, text_i) / τ)) ]   (image→text direction)
  + -1/N × Σᵢ [ log(softmax(sim(text_i, img_i) / τ)) ]   (text→image direction)

where τ (temperature) ≈ 0.07 — controls sharpness
```

**Key insight:** Larger batch = harder negatives (more wrong pairs to distinguish from). CLIP uses batches of 32,768. This is why it requires massive compute.

```python
import torch
import torch.nn.functional as F

def clip_loss(image_embeds, text_embeds, temperature=0.07):
    """
    image_embeds: (N, D) — L2-normalized image embeddings
    text_embeds:  (N, D) — L2-normalized text embeddings
    """
    # Similarity matrix: (N, N)
    logits = (image_embeds @ text_embeds.T) / temperature

    # Labels: diagonal is the correct pair
    labels = torch.arange(len(image_embeds), device=image_embeds.device)

    # Symmetric cross-entropy loss
    loss_img  = F.cross_entropy(logits,   labels)  # image → text
    loss_text = F.cross_entropy(logits.T, labels)  # text  → image
    return (loss_img + loss_text) / 2
```

---

## 4. Zero-Shot Classification

CLIP can classify images without any task-specific training:

```
Task: "Is this image a sneaker, dress, or bag?"

Step 1: Encode label texts
  "a photo of a sneaker"  → text_embed_1
  "a photo of a dress"    → text_embed_2
  "a photo of a bag"      → text_embed_3

Step 2: Encode query image
  image → img_embed

Step 3: Find closest text
  scores = cosine_sim(img_embed, [text_embed_1, text_embed_2, text_embed_3])
  prediction = argmax(scores)
```

**No training data for the target task — just describe the classes in natural language.**

This beat supervised ResNet-50 on ImageNet with zero task-specific training, which was the original paper's key result.

---

## 5. CLIP Variants & Related Models

### OpenCLIP (Open Source Reproduction)
- Trained on LAION-400M and LAION-2B (publicly available datasets)
- Multiple model sizes; matches or exceeds OpenAI CLIP quality
- License: MIT — usable in production without restrictions
- **Use this if you can't use the OpenAI CLIP API**

### SigLIP (Google, 2023)
- Replaces InfoNCE with a **sigmoid loss** — each pair is scored independently
- No need for large batch size (works with batch=256 vs CLIP's 32K)
- Faster training, comparable or better quality
- Used in Google's Gemini multimodal models

```
InfoNCE:  Softmax across all negatives in batch — needs huge batches
SigLIP:   Sigmoid per pair — works with small batches, scales better
```

### ALIGN (Google, 2021)
- 1.8 billion (image, alt-text) pairs — noisier but much larger than CLIP
- Same contrastive approach
- Result: scale beats curation quality

### CoCa (Contrastive Captioners, Google, 2022)
- CLIP + captioning loss combined
- Image encoder: contrastive loss
- Image-Text decoder: autoregressive captioning loss
- Single model handles retrieval AND generation

### EVA-CLIP (BAAI, 2023)
- Scaling CLIP to ViT-18B (18 billion parameters)
- State-of-the-art on many vision benchmarks
- Shows CLIP quality keeps improving with scale

### DINOv2 (Meta, 2023) — CLIP's Self-Supervised Cousin
- No text — pure vision self-supervision
- Better than CLIP on dense visual tasks (segmentation, depth)
- Often combined with CLIP: CLIP for semantics, DINOv2 for fine-grained visual detail

---

## 6. Fine-Tuning CLIP

Pretrained CLIP has **domain shift** for specialized applications (medical, satellite, product images). Fine-tuning adapts it.

### Option A: Full Fine-Tuning
- Update all weights of both encoders
- Requires large dataset (>1M pairs) — otherwise catastrophic forgetting
- Best quality

### Option B: Linear Probing
- Freeze both encoders, train only a classification head on top
- Fastest, minimal compute, works with small datasets
- Lower ceiling than full fine-tuning

### Option C: LoRA / Adapter Fine-Tuning (Recommended)
- Insert small trainable adapter layers into frozen encoder
- Train only adapters (~1% of total params)
- No catastrophic forgetting
- Works well with 10K–100K domain-specific pairs

```python
class CLIPAdapter(torch.nn.Module):
    """
    Lightweight adapter inserted into frozen CLIP layers.
    Only this module is trained; CLIP backbone is frozen.
    """
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.down = torch.nn.Linear(dim, dim // reduction)
        self.up   = torch.nn.Linear(dim // reduction, dim)
        self.scale = torch.nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        return x + self.scale * self.up(F.relu(self.down(x)))
```

### Option D: Prompt Tuning (CoCoOp, 2022)
- Learn soft text prompts instead of "a photo of a [class]"
- Only prompt embeddings are trained (a few hundred parameters)
- Effective for few-shot adaptation

```
Standard:  "a photo of a [sneaker]"   ← hard-coded text template
CoCoOp:    [v1][v2][v3] "a [sneaker]" ← v1,v2,v3 are learned vectors
```

---

## 7. Use Cases

### Image-to-Image Retrieval
- Encode a query image → find products/images with closest embedding in FAISS index
- Core of Pinterest Lens, Google Lens, Amazon StyleSnap

### Text-to-Image Search
- Encode a text query → retrieve images with closest embedding
- Powers Unsplash search, Getty Images AI search, DALL-E prompt pre-filtering

### Cross-Modal E-commerce Search
```
User types: "casual linen shirt for summer"
  → text embed → ANN search in product image index
  → Returns visually matching products with no keyword dependency

User uploads a photo:
  → image embed → ANN search in product text/title index
  → Returns matching titles even with no visual metadata
```

### Zero-Shot Image Classification
- Medical imaging: classify X-rays, pathology slides using clinical text descriptions
- Satellite imagery: detect land use patterns with text labels
- Manufacturing QA: detect defects described in natural language

### Vision-Language Foundation for Downstream Models
- CLIP image encoder as a frozen backbone for many tasks:
  - Image captioning (feed CLIP features to a language model)
  - Visual QA
  - Image segmentation (CLIPSeg)
  - Object detection (GLIP, Grounding DINO)

### Content Moderation
- Detect policy-violating content without training per-category classifiers
- Add new violation categories by writing a text description — zero-shot

---

## 8. Limitations

| Limitation | Details |
|---|---|
| **Fine-grained counting** | "Two dogs" vs "three dogs" — text encoder poorly represents numbers |
| **Spatial relationships** | "dog on the left of cat" — CLIP struggles with spatial text |
| **OCR / text in images** | Reading text in an image is not what CLIP was trained for |
| **Rare / niche concepts** | Web data skews common concepts; rare items may have weak embeddings |
| **Compositionality** | "A red car next to a blue truck" — each attribute OK, combined scene harder |
| **Adversarial vulnerability** | Small image perturbations can flip the embedding |

---

## 9. Interview Checkpoint

1. **"Explain how CLIP learns without task-specific labels."**
   - Contrastive pretraining on (image, text) pairs. The loss pulls matching pairs together and pushes non-matching pairs apart. The text descriptions act as free supervision.

2. **"What is temperature τ in the InfoNCE loss and why does it matter?"**
   - Controls how sharply peaked the softmax distribution is. Low τ → hard, confident predictions; too low causes training instability. High τ → soft distribution; model learns less. Typically 0.07 (learned or fixed).

3. **"Why does CLIP need such large batch sizes?"**
   - Each batch provides N² potential negatives. Larger batches = harder, more diverse negatives = better learning signal. CLIP uses 32,768 — the original paper shows quality degrades significantly with smaller batches.

4. **"When would you choose DINOv2 over CLIP for image search?"**
   - DINOv2 for image-only retrieval where fine-grained visual detail matters (exact product match, texture). CLIP when you need multi-modal (text + image) queries or zero-shot generalization to unseen categories.

5. **"How would you adapt CLIP for a medical imaging use case?"**
   - Start with BiomedCLIP (already fine-tuned on medical literature + images). If not available, fine-tune with LoRA adapters using (X-ray, radiology report) pairs. Evaluate with AUC on held-out diagnoses, not just retrieval recall.
