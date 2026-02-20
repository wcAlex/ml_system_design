# 05 — Feature Engineering

## Beyond Raw Embeddings

The image embedding is the core feature, but a production system uses a rich set of additional features — for re-ranking, filtering, and personalization. Feature engineering is where domain knowledge creates value that pure ML cannot capture.

---

## Feature Taxonomy

```
Features
├── Visual Features
│   ├── Global embedding (dense vector from encoder)
│   ├── Regional embeddings (per object, per region)
│   ├── Attribute predictions (color, texture, style)
│   └── Perceptual features (color histogram, HOG)
├── Product Features
│   ├── Category / taxonomy
│   ├── Price tier
│   ├── Brand embedding
│   ├── Seller quality score
│   └── Product age / freshness
├── Context Features
│   ├── User device (mobile vs desktop)
│   ├── Time of day / season
│   └── Geographic location
└── User Features (for personalization — see doc 13)
    ├── Long-term preference embedding
    ├── Session history
    └── Purchase history
```

---

## Visual Features

### Feature 1: Global Embedding (Primary)

The output of the encoder model (512-d or 768-d vector). This captures holistic visual semantics — style, shape, color overall.

```
Image → Encoder (CLIP ViT-B/16) → [CLS] token → 512-d L2-normalized vector
```

**Cosine similarity** between query and product embeddings is the primary retrieval signal.

### Feature 2: Attribute Predictions (Structured)

Train a **multi-task classifier** on top of the embedding to predict structured attributes:

| Attribute | Type | Example Values |
|---|---|---|
| Category | Multi-class | sneakers, boots, sandals |
| Color | Multi-label | red, white, black |
| Pattern | Multi-class | solid, striped, floral |
| Material | Multi-class | leather, canvas, suede |
| Style | Multi-class | casual, formal, athletic |
| Gender | Multi-class | men, women, unisex |

**Why add attribute features?**
- Allow **hard filtering** — "show me only red sneakers" even if embedding captures red+sneaker separately
- Improve **interpretability** — you can explain why an item was returned
- Used in **re-ranking** — attribute match boosts score

**Architecture:**
```
Frozen encoder backbone
        │
        ▼
  512-d embedding
        │
   ┌────┴─────┬──────────┬──────────┐
   ▼          ▼          ▼          ▼
Color head  Category  Pattern    Gender
(softmax)   (softmax) (softmax)  (sigmoid)
```

Train with multi-task loss: `L = L_embedding + λ₁L_color + λ₂L_category + ...`

### Feature 3: Color Histogram (Lightweight, Rule-Based)

```python
# Extract dominant colors using k-means on pixel values
from sklearn.cluster import KMeans
import numpy as np

def dominant_colors(image_rgb, k=5):
    pixels = image_rgb.reshape(-1, 3).astype(float)
    kmeans = KMeans(n_clusters=k, n_init=3)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_  # k×3 matrix of RGB centroids
```

**Use:** Color-based hard filtering. If query image is predominantly red and user explicitly filters "red items," use this as a post-retrieval filter.

**Pros:** Fast, explainable, no ML needed.
**Cons:** Semantic colors ("nude," "camel") don't map cleanly to RGB.

### Feature 4: Regional Embeddings (Advanced)

Instead of one global embedding, extract embeddings for detected regions.

```
Image
  │
  ▼ Object detection (YOLO / Detectron2)
[bounding box 1: dress] [bounding box 2: shoes] [bounding box 3: bag]
  │                        │                       │
  ▼                        ▼                       ▼
embedding_1             embedding_2             embedding_3
```

**Use case:** A user uploads a full outfit photo. You want to match just the shoes, not the whole outfit. Regional embeddings let you search by product-level region.

**Industry example:** Pinterest Lens allows users to select a crop. Google Lens automatically detects product regions.

**Complexity:** High — requires object detection model + region pooling. Worth it for fashion, home décor; less critical for single-item queries.

---

## Product Features

### Metadata Embedding

Category labels and brand names are **categorical features** that can be embedded:

```python
# Category embedding: learn 64-d embeddings for each category
# Similar to word embeddings — "sneakers" and "running shoes" should be close
category_embedding = nn.Embedding(num_categories, 64)
brand_embedding = nn.Embedding(num_brands, 32)
```

These are concatenated with the visual embedding in the re-ranking model:

```
[visual_embedding (512) | category_embedding (64) | brand_embedding (32) | price_bucket (8)]
                                       │
                              Re-ranking MLP
```

### Price Tier Feature

Bucket price into tiers: budget (<$50), mid ($50–$200), premium ($200–$500), luxury (>$500)

- One-hot or ordinal encoding
- Use to penalize results far from query item's inferred price point (if available)

### Recency / Freshness

Products added recently should get a small boost (new inventory):
```
freshness_score = exp(-days_since_added / 30)
```

Products older than 6 months may be deprioritized unless evergreen.

---

## Feature Stores

In production, precomputed features must be stored and retrieved efficiently.

### Option 1: Offline Feature Store (Batch)
- Product embeddings precomputed daily/hourly
- Stored in columnar format (Parquet) or key-value store (Redis)
- Retrieved at re-ranking time by product ID

### Option 2: Online Feature Store (Low-Latency)
- Tools: Feast, Tecton, Vertex AI Feature Store
- Real-time features (user session data, inventory) retrieved at query time
- P99 < 5ms lookup latency

### Option 3: Feature-in-Index (Simple, Efficient)
- Store all product features alongside the ANN index
- FAISS supports payload storage; Milvus has native metadata storage
- Pros: one lookup for embedding + metadata; Cons: index becomes large

---

## Feature Interactions at Re-ranking

The re-ranking stage combines all features into a score:

```
score(query, product) =
    α × cosine_sim(q_embed, p_embed)         # visual similarity
  + β × attribute_match(q_attrs, p_attrs)    # color/style match
  + γ × personalization_score(user, product) # preference match
  + δ × seller_quality_score(product)        # business rule
  + ε × freshness_score(product)             # recency bonus
  - ζ × price_distance(query_price, p_price) # price relevance
```

Weights (α, β, γ...) are tunable, can be learned with a simple linear model, or optimized via online A/B experiments.

---

## Interview Checkpoint

1. **"How do you handle color search when embedding alone is insufficient?"**
   - Combine embedding similarity with color histogram distance. Or train a dedicated color classification head and filter by predicted color attribute.

2. **"What is the difference between features used in retrieval vs re-ranking?"**
   - Retrieval features: must be low-dimensional, precomputed, support fast ANN (embedding only). Re-ranking features: richer, can include user features and metadata, computed at query time on a small candidate set (~500 items).

3. **"How do you ensure feature consistency between training and serving?"**
   - Feature store ensures the same preprocessing pipeline runs offline (training) and online (serving). A training-serving skew is a common source of model degradation.

4. **"Should you L2-normalize embeddings before storing in the index?"**
   - Yes, for cosine similarity — L2-normalize makes cosine similarity equivalent to dot product, which FAISS and ScaNN optimize for with IP (inner product) distance. Don't renormalize at query time.
