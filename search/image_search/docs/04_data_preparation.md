# 04 — Data Preparation

## Data is the Foundation

The quality of your embeddings is bounded by your training data. This section covers where data comes from, how to label it, and how to prepare it for training.

---

## Data Sources

### Source 1: Product Catalog (Core)
**What you have naturally:**
- Product images (multiple angles per SKU)
- Product title, description, category taxonomy
- Attributes: color, material, brand, size
- Price, seller, inventory

**Volume:** 10M–100M products for large e-commerce platforms
**Quality:** High (controlled studio photos) — but also user-uploaded (noisy)

**Pairs you can generate for free:**
- (image_A, image_B) are positive pairs if they belong to the same product (different angles)
- (image, product_title) pairs for CLIP-style training

### Source 2: User Interaction Logs (Behavior Signal)
**What you collect:**
- Search queries + clicks (click = soft positive)
- Purchases (purchase = strong positive, but sparse ~1–3%)
- "Similar items" page: if user views item A then clicks item B, A and B are similar
- Add-to-cart without purchase = weak positive

**Volume:** Millions of events per day on large platforms

**Caution:** Behavioral data has **popularity bias** — popular items get more clicks regardless of actual visual similarity. Must be debiased.

### Source 3: Human-Labeled Pairs (Ground Truth)
**What:** Human raters label (query image, product) as relevant/not relevant
**Volume:** Typically 10K–100K pairs (expensive to scale)
**Use:** Offline evaluation (not training — too small)

**Grading rubric for raters:**
- **4 (Exact):** Same product, different photo
- **3 (Highly Similar):** Same style, different color
- **2 (Similar):** Same category, similar style
- **1 (Related):** Same category, different style
- **0 (Irrelevant):** Wrong category

### Source 4: Synthetic Data
**Techniques:**
- Augmentation (covered below)
- Generative AI: use diffusion models to create synthetic product images (emerging practice at Amazon, eBay)
- Style transfer: render same product in different backgrounds/colors

---

## Data Labeling Strategies

### Option 1: Manual Annotation (Gold Standard)
- Human raters label (query, product) relevance
- **Pros:** Highest quality, needed for evaluation
- **Cons:** Expensive ($2–5/pair), slow to scale

### Option 2: Catalog-Derived Labels (Programmatic)
- Same SKU = positive pair; different category = negative pair
- **Pros:** Free, scalable to millions of pairs
- **Cons:** Same category ≠ always visually similar (a red dress ≠ a black dress)

### Option 3: Behavior-Derived Labels (Click-Through)
- (query_image, clicked_product) = positive pair
- (query_image, skipped_product_shown_above_click) = hard negative
- **Pros:** Captures user intent directly
- **Cons:** Noisy (accidental clicks), biased toward popular items

### Option 4: Cross-Modal Self-Supervision (for CLIP)
- (product_image, product_title) = natural positive pair
- No labeling needed — just collect catalog data
- **Pros:** Scales to billions of pairs
- **Cons:** Title may not perfectly describe the visual appearance

**Best practice:** Use a combination — catalog pairs for training, behavior signals for hard negative mining, human labels for evaluation.

---

## Data Pipeline Architecture

```
Raw Sources                Processing                  Training Dataset
──────────────────────────────────────────────────────────────────
Product DB ──────────────► Image downloader ──────────► Image store (S3)
                               │
                               ▼
User logs (Kafka) ──────────► Event processor ──────────► Click pairs table
                               │
                               ▼
Human labels ───────────────► Label store ─────────────► Eval set
                                                              │
                           Feature Store                      ▼
                        (precomputed embeddings)         Training batches
                                                         (Triplets / Pairs)
```

---

## Image Preprocessing

### Standard Pipeline (Per Image)
```python
# Standard preprocessing for ViT/CLIP
transforms = [
    Resize(224, 224),          # or 336x336 for CLIP ViT-L/14
    CenterCrop(224),            # remove borders
    ToTensor(),                 # HWC uint8 -> CHW float32
    Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet mean (also good for CLIP)
        std=[0.229, 0.224, 0.225]
    )
]
```

### Handling Real-World Query Images (User-Uploaded)
User photos are messier than catalog photos:

| Issue | Handling |
|---|---|
| Wrong orientation | EXIF rotation correction |
| Multiple objects | Object detection (YOLO) + largest box crop |
| Extreme blur | Blur score (Laplacian variance) — reject if < threshold |
| Low resolution | Upsample with ESRGAN if < 64×64 |
| NSFW content | NSFW classifier before embedding |

---

## Data Augmentation (Training Only)

Augmentation artificially expands your training set and teaches the model **invariances** — what should NOT change the embedding.

| Augmentation | Invariance Learned | Strength |
|---|---|---|
| **Random crop + resize** | Scale, framing | Strong |
| **Horizontal flip** | Mirror symmetry | Medium |
| **Color jitter** (brightness, contrast, saturation) | Lighting conditions | Medium |
| **Gaussian blur** | Focus/depth of field | Weak |
| **Random grayscale** | Color → texture focus | Context-dependent |
| **Random erasing** | Occlusion robustness | Medium |
| **Cutmix / Mixup** | Interpolation regularization | Weak for retrieval |

**Do NOT apply:** Geometric distortions (rotation, shear) for product images — products have canonical orientations.

**CLIP-specific:** CLIP's pretraining already learns many invariances. Use lighter augmentation (random crop + flip + light color jitter) for fine-tuning.

---

## Negative Mining (Critical for Metric Learning)

The quality of negatives determines embedding quality. Hard negatives = similar-looking items that are different — these force the model to learn fine-grained distinctions.

### Option 1: Random Negatives
- Sample random products as negatives
- **Pros:** Simple
- **Cons:** Too easy — "red sneaker" vs "blue umbrella" — model learns nothing fine-grained

### Option 2: In-Batch Hard Negatives
- Within each training batch, items from different SKUs serve as negatives
- Most negative pairs in a batch are "hard" simply by construction
- **Pros:** Efficient, no extra mining step
- **Cons:** Batch size limits the hardness

### Option 3: Offline Hard Negative Mining
- After each epoch, compute embeddings for all training items
- Find approximate nearest neighbors that are NOT positive pairs → hard negatives
- **Pros:** Very hard negatives; dramatically improves embedding quality
- **Cons:** Expensive (requires full embedding pass), can lead to false negatives

**Industry practice:** Use **in-batch negatives** as the base, add **offline hard negative mining** once a week or epoch-level. Pinterest, Google, and Meta all use some form of hard negative mining.

---

## Data Quality Issues to Address

| Issue | Impact | Fix |
|---|---|---|
| Duplicate products | False negatives in training | Dedup by image hash + title similarity |
| Label noise (wrong category) | Corrupted training signal | Confident learning: filter low-confidence labels |
| Popularity bias | Popular items get too-close embeddings | Re-sample based on popularity; debiasing loss |
| Distribution shift (catalog vs. user photos) | Model fails on real queries | Add user-uploaded images to training set |
| Stale embeddings | New products not searchable | Hourly incremental embedding updates |

---

## Interview Checkpoint

1. **"How do you create training pairs without human labels?"**
   - Same-SKU different-angle = positive. Different category = easy negative. Same category, different brand = hard negative via mining.

2. **"How do you deal with class imbalance? (Millions of clothing items, few furniture items)"**
   - Stratified sampling per category. Over-sample rare categories in training batches.

3. **"What is a false negative and why does it matter in contrastive training?"**
   - A false negative is a pair labeled as "negative" but is actually visually similar (e.g., two different SKUs that look identical). It pushes similar items apart, hurting embedding quality. Mitigation: dedup catalog, use high-confidence negatives.

4. **"How do you handle the cold-start problem for new products?"**
   - New products get embeddings generated at index time by the current model (no retraining needed). The model generalizes. Only retrain to adapt to new product categories or seasonal trend shifts.
