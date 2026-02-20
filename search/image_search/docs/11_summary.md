# 11 — System Summary

## The Big Picture

This document ties everything together: the architecture, components, data flows, and key design decisions.

---

## Full Architecture Diagram

```
═══════════════════════════════════════════════════════════════════════════════
                        IMAGE SEARCH SYSTEM — E-COMMERCE
═══════════════════════════════════════════════════════════════════════════════

 ╔══════════════════════════ OFFLINE PIPELINE ═══════════════════════════════╗
 ║                                                                           ║
 ║  ┌─────────────┐     ┌────────────────┐     ┌──────────────────────────┐ ║
 ║  │ Product DB  │────►│ Image Download │────►│ GPU Batch Embedding Job  │ ║
 ║  │ (catalog)   │     │ + QC filter    │     │ CLIP ViT-B/16            │ ║
 ║  └─────────────┘     └────────────────┘     │ Batch: 512 imgs/step     │ ║
 ║                                             │ Output: 512-d vectors    │ ║
 ║  ┌─────────────┐                            └──────────┬───────────────┘ ║
 ║  │ User Click  │     ┌────────────────┐               │                  ║
 ║  │ Logs (Kafka)│────►│ Hard Negative  │               ▼                  ║
 ║  └─────────────┘     │ Mining Job     │     ┌──────────────────────────┐ ║
 ║                      └────────┬───────┘     │ Embedding Store (S3)     │ ║
 ║  ┌─────────────┐              │             │ + Feature Store (Redis)  │ ║
 ║  │ Human Labels│     ┌────────▼───────┐     └──────────┬───────────────┘ ║
 ║  │ (eval only) │────►│ Model Training │               │                  ║
 ║  └─────────────┘     │ (InfoNCE loss) │               ▼                  ║
 ║                      └────────────────┘     ┌──────────────────────────┐ ║
 ║                                             │ FAISS Index Builder      │ ║
 ║                                             │ IVFFlat (fast) or        │ ║
 ║                                             │ IVFPQ (memory-efficient) │ ║
 ║                                             └──────────┬───────────────┘ ║
 ╚══════════════════════════════════════════════════════════╪════════════════╝
                                                            │
                                                   Nightly rebuild
                                                   Hourly incremental
                                                            │
 ╔═════════════════════════ ONLINE PIPELINE ═════════════════╪════════════════╗
 ║                                                           │               ║
 ║  User uploads photo                              ┌────────▼────────┐     ║
 ║       │                                          │  FAISS Index    │     ║
 ║       ▼                                          │ (10M products)  │     ║
 ║  ┌──────────────┐                                │ nprobe=50       │     ║
 ║  │ API Gateway  │                                └────────▲────────┘     ║
 ║  │ - Auth       │                                         │              ║
 ║  │ - Rate limit │                                         │              ║
 ║  │ - Validation │                                         │              ║
 ║  └──────┬───────┘                                         │              ║
 ║         │                                                 │              ║
 ║         ▼                                                 │              ║
 ║  ┌──────────────┐     ┌──────────────────────┐           │              ║
 ║  │  Preprocess  │────►│ Query Embedding Svc  │───────────┘              ║
 ║  │ - Resize     │     │ CLIP ViT-B/16 (GPU)  │  512-d query vector      ║
 ║  │ - EXIF fix   │     │ ~20ms latency        │                           ║
 ║  │ - NSFW check │     └──────────────────────┘                           ║
 ║  └──────────────┘                                                         ║
 ║                              ANN search returns top-500 candidates        ║
 ║                                                 │                         ║
 ║                                                 ▼                         ║
 ║  ┌────────────────────────────────────────────────────────────────────┐  ║
 ║  │                     RE-RANKING SERVICE                             │  ║
 ║  │                                                                    │  ║
 ║  │  Hard Filters          Attribute Match     Personalization         │  ║
 ║  │  - in_stock            - color             - user history          │  ║
 ║  │  - price range         - category          - session context       │  ║
 ║  │  - category filter     - style             - purchase signal       │  ║
 ║  │           │                   │                    │               │  ║
 ║  │           └───────────────────┴────────────────────┘               │  ║
 ║  │                               │                                    │  ║
 ║  │                          Final Score                               │  ║
 ║  │              α·visual + β·attr + γ·pers + δ·fresh                 │  ║
 ║  └──────────────────────────────┬─────────────────────────────────────┘  ║
 ║                                 │                                         ║
 ║                                 ▼                                         ║
 ║                           Top-20 Results                                  ║
 ║                           returned to user                                ║
 ║                                 │                                         ║
 ║                                 ▼                                         ║
 ║                    ┌────────────────────────┐                            ║
 ║                    │ Logging + Feedback Loop│                            ║
 ║                    │ - Query embedding      │                            ║
 ║                    │ - Clicks/purchases     │                            ║
 ║                    │ → Training data store  │                            ║
 ║                    └────────────────────────┘                            ║
 ╚═════════════════════════════════════════════════════════════════════════════╝
```

---

## Component Map

| Component | Technology | Purpose |
|---|---|---|
| **Image embedding model** | CLIP ViT-B/16 (fine-tuned) | Core visual understanding |
| **ANN index** | FAISS IVFFlat / IVFPQ | Fast approximate nearest neighbor search |
| **Inference server** | Triton / TorchServe + GPU | Low-latency model serving |
| **Embedding store** | S3 + Parquet | Offline embedding storage and re-use |
| **Feature store** | Redis + Feast | Real-time feature retrieval for re-ranking |
| **Stream processing** | Kafka | User event ingestion, new product pipeline |
| **Batch processing** | Spark / Ray | Batch embedding jobs, hard negative mining |
| **Model training** | PyTorch + InfoNCE | Contrastive fine-tuning |
| **Experiment tracking** | MLflow / W&B | Metric tracking, model versioning |
| **API gateway** | Kong / AWS API GW | Auth, rate limiting, routing |
| **CDN** | CloudFront / Fastly | Static asset caching |

---

## Data Flow Summary

```
TRAINING DATA FLOW:
Product catalog images
  → GPU batch embedding (CLIP)
    → Contrastive pairs (same-SKU positives, hard negatives via mining)
      → InfoNCE training
        → Fine-tuned CLIP model
          → New embeddings for all products
            → FAISS index rebuild

SERVING DATA FLOW:
User photo upload
  → Preprocessing (resize, EXIF, quality check)
    → CLIP inference (query embedding)
      → FAISS ANN search (top-500 candidates)
        → Re-ranking (filters + personalization + business rules)
          → Top-20 results → user
            → Click/purchase events → training log
```

---

## Key Design Decisions and Rationale

| Decision | Choice | Why |
|---|---|---|
| Embedding model | CLIP ViT-B/16 fine-tuned | Multi-modal, strong generalization, upgradeable |
| Training loss | InfoNCE (contrastive) | More efficient than triplet; uses all in-batch negatives |
| ANN index | FAISS IVFFlat | Well-tested, flexible, controllable recall/speed trade-off |
| Index freshness | Nightly rebuild + hourly incremental | Balances freshness vs. complexity |
| Retrieval → re-ranking split | 500 → 20 (25x reduction) | Allows rich re-ranking features without ANN cost |
| Embedding dimension | 512-d | Balance: quality vs. index size vs. ANN speed |
| Latency budget | 200ms P99 | Industry standard for search UX |

---

## What We're NOT Building (Simplifications)

For a real production system at Google/Amazon scale, you'd also add:

- **Federated learning** — privacy-preserving personalization
- **Online learning** — real-time embedding updates from click feedback
- **Multi-region deployment** — geo-distributed serving for global latency
- **Model compression pipeline** — automated distillation + quantization
- **Adversarial robustness** — handle adversarial query images (rare, but relevant for fashion fraud)
- **Catalog graph** — link products by brand, style, outfit compatibility
- **Video search** — extend to frame-level video indexing

---

## System Health Monitoring

Key metrics to monitor in production:

```
Serving health:
  - Embedding service P50/P95/P99 latency
  - ANN service P50/P95/P99 latency
  - End-to-end P99 latency
  - QPS (requests per second)
  - Error rate (4xx, 5xx)

Search quality:
  - Daily CTR (7-day rolling)
  - Conversion rate (7-day rolling)
  - Search abandonment rate
  - Zero-result query rate (no results returned)

Model health:
  - Embedding distribution drift (cosine similarity to reference set)
  - Index staleness (time since last full rebuild)
  - Fraction of products with embeddings (should be ~100%)
```

**Alert thresholds:** Page on-call if P99 latency > 300ms, error rate > 1%, or CTR drops > 5% day-over-day.

---

## Interview Summary: Key Points to Hit

When explaining this system in an interview, hit these 5 points:

1. **Two-stage pipeline:** ANN retrieval (fast, coarse) → re-ranking (slow, fine). This is the universal pattern for any large-scale search.

2. **Embedding quality is everything:** The ANN index is just a lookup. The model quality determines relevance ceiling.

3. **Offline-online gap:** What works in training often doesn't work in production. Address distribution shift explicitly.

4. **Freshness vs. cost:** Index rebuild frequency is a business + engineering trade-off. Know the options.

5. **Metrics matter:** Know which metric you're optimizing at each stage (Recall@K for retrieval, NDCG for ranking, CTR/conversion online).
