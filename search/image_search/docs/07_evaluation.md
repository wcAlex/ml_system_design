# 07 — Evaluation

## Two-Track Evaluation: Offline + Online

A model that scores well offline but fails online is the most dangerous outcome. Always run both tracks.

---

## Offline Evaluation (Pre-deployment)

### Evaluation Dataset Setup

**Gold set construction:**
- 5,000–50,000 (query image, relevant products) pairs, human-labeled
- Each query has graded relevance: 0–4 scale (see doc 04)
- Stratified across categories (don't over-represent shoes if catalog is balanced)

**Split:**
- Train / Val / Test — but keep the test set held out until final model evaluation
- Temporal split is better than random split for production: train on older data, test on recent data

---

## Retrieval Metrics

### Recall@K (Most Important for ANN Stage)

```
Recall@K = |Relevant items in top-K| / |All relevant items|
```

| K | Meaning | Target |
|---|---|---|
| Recall@1 | Exact match at top | Precision-focused use case |
| Recall@10 | At least one relevant in top 10 | Standard ANN evaluation |
| Recall@100 | At least one relevant in top 100 | Measures retrieval coverage |

**Why Recall@K for ANN?** ANN retrieval is about getting the right candidates into your top-K pool. Re-ranking handles the ordering. So recall is more important than precision at this stage.

**Target:** Recall@10 ≥ 0.80 (strong), ≥ 0.90 (excellent)

### Precision@K

```
Precision@K = |Relevant items in top-K| / K
```

More useful for the re-ranked final results returned to the user.

### NDCG@K (Normalized Discounted Cumulative Gain)

Best metric when you have **graded relevance** (0–4 scale):

```
DCG@K = Σᵢ₌₁ᴷ (2^relevance_i - 1) / log₂(i + 1)
NDCG@K = DCG@K / IDCG@K   (IDCG = ideal ordering)
```

Higher-ranked results count more. A highly relevant item at rank 1 is better than rank 5.

**Target:** NDCG@10 ≥ 0.60 (acceptable), ≥ 0.75 (strong)

### MRR (Mean Reciprocal Rank)

```
MRR = (1/|Q|) × Σ_q (1 / rank of first relevant result)
```

Focuses on: how early does the first relevant result appear?

**Target:** MRR ≥ 0.50 for fashion visual search

### mAP (Mean Average Precision)

Average of precision at each relevant item's rank. Good for comprehensive evaluation across all relevant items.

### Metric Selection for Interviews

| Scenario | Primary Metric | Rationale |
|---|---|---|
| ANN retrieval quality | Recall@10 | Are candidates correct? |
| Re-ranked results | NDCG@10 | Is ordering correct? |
| Single best result matters | MRR | Is rank 1 right? |
| Multiple relevant items | mAP | Are all relevant items found? |

---

## Embedding Quality Metrics (Intrinsic Evaluation)

Before checking retrieval metrics, validate the embedding space itself:

### Embedding Distribution Check
```python
import numpy as np

def check_embedding_health(embeddings):
    """
    embeddings: (N, D) matrix, L2-normalized
    """
    norms = np.linalg.norm(embeddings, axis=1)
    sim_matrix = embeddings @ embeddings.T

    print(f"Norm mean: {norms.mean():.4f} (should be ~1.0 after L2 norm)")
    print(f"Intra-cluster similarity mean: {sim_matrix.mean():.4f}")
    print(f"Embedding variance per dim: {embeddings.var(axis=0).mean():.4f}")

    # Collapse detection: if std is near 0, embeddings have collapsed
    if embeddings.std() < 0.01:
        print("WARNING: Embedding collapse detected!")
```

### k-NN Precision at Class Level
- For a sample of products, check if k nearest neighbors are in the same category
- **Category-level kNN-precision@5** ≥ 0.85 is a strong signal

---

## Online Evaluation (Post-deployment A/B Test)

### A/B Test Setup
- **Control:** Current production model (e.g., pretrained CLIP)
- **Treatment:** Fine-tuned CLIP (new candidate)
- **Traffic split:** 50/50 or 90/10 (conservative rollout)
- **Duration:** 2 weeks minimum (capture weekly seasonality)
- **Sample size:** Use power analysis to determine min users per group

### Online Metrics

| Metric | Formula | Target |
|---|---|---|
| **CTR** | Clicks / Search impressions | Treatment CTR > Control CTR |
| **Conversion Rate** | Purchases / Search sessions | Primary business metric |
| **Time to First Click** | Seconds from results shown to first click | Proxy for result quality |
| **Search Reformulation Rate** | Sessions with query reformulation / total sessions | Low = found what they wanted |
| **Search Abandonment** | Sessions with no click / total sessions | Low = results were satisfying |
| **Revenue per Session** | GMV / Search sessions | Business health |

### Statistical Significance
- **p-value < 0.05** before claiming a win
- **Minimum detectable effect (MDE):** typically 1–2% CTR lift is meaningful
- **Multi-metric testing:** Use Bonferroni correction to avoid false positives across metrics

---

## Evaluation Anti-patterns (Common Mistakes)

| Anti-pattern | Problem | Fix |
|---|---|---|
| Testing on training data | Wildly optimistic metrics | Always use held-out test set |
| Random test/train split (no temporal) | Leakage from future data | Use time-based split |
| Only evaluating on popular queries | Miss tail failure modes | Stratify evaluation by query frequency |
| Not checking calibration | Scores don't reflect actual similarity | Check score distribution and rank correlation |
| Comparing models on different eval sets | Apples to oranges | Fix eval set for all model comparisons |
| Shipping based on offline metrics only | Offline-online gap | Always A/B test before full rollout |

---

## Evaluation Pipeline Architecture

```
New Model Checkpoint
      │
      ▼ (automated CI/CD)
┌─────────────────────────────┐
│  Offline Eval Job           │
│  - Compute embeddings       │
│  - ANN search on eval set   │
│  - Compute Recall@K, NDCG   │
│  - Log to MLflow / W&B      │
└─────────────┬───────────────┘
              │ Pass threshold?
              ▼
┌─────────────────────────────┐
│  Shadow Mode Testing        │
│  - Run new model in shadow  │
│  - Compare results offline  │
│  - No user impact           │
└─────────────┬───────────────┘
              │ Looks good?
              ▼
┌─────────────────────────────┐
│  A/B Test (5% traffic)      │
│  - Monitor CTR, conversion  │
│  - Run 2 weeks              │
└─────────────┬───────────────┘
              │ Significant lift?
              ▼
         Full Rollout
```

---

## Interview Checkpoint

1. **"How do you decide which metric to optimize for?"**
   - It depends on the business goal. If the product is browsing/inspiration (Pinterest), optimize NDCG (ranking quality). If it's purchase intent (Amazon), optimize CTR and conversion. Never optimize multiple metrics without prioritization.

2. **"Offline metrics improved but online metrics didn't — what's wrong?"**
   - Likely causes: (1) evaluation set doesn't match real query distribution; (2) model improved for easy queries but degraded for tail queries; (3) serving latency regression causing user drop-off; (4) feature skew between training and serving.

3. **"How long do you run an A/B test?"**
   - Minimum 2 weeks to capture weekly seasonality effects. Use a power calculator: with 5% MDE at 80% power and α=0.05, you need N users per group. Most large e-commerce platforms can hit this in days, but still run for 2 weeks.

4. **"What do you do if conversion is neutral but CTR improves?"**
   - Investigate — users may be clicking but not finding purchase-worthy items (result quality issue). Or the new model returns items in different price ranges. Don't ship on CTR alone; conversion is the ground truth.
