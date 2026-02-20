# 01 — ML Objectives

## The First Question in Any Interview: What Are We Optimizing For?

Before writing a single line of code, you must translate a business goal into a measurable ML objective. Getting this wrong is the most common failure point.

---

## Business Context

**Product:** E-commerce visual search (like Amazon StyleSnap, Pinterest Shopping Lens)

**User story:** "I saw someone wearing cool sneakers on the street. I take a photo and want to buy them (or something similar)."

**Stakeholders and what they care about:**
| Stakeholder | Goal |
|---|---|
| User | Find what they're looking for quickly |
| Business | Increase conversion, GMV (gross merchandise value) |
| Seller | Get their products discovered |
| Platform | Keep users engaged, returning |

---

## Business Metrics (North Star)

These are what the business cares about. ML must serve these.

| Metric | Definition | Why it matters |
|---|---|---|
| **Conversion Rate** | Purchases / Search sessions | Direct revenue signal |
| **GMV from visual search** | Revenue attributed to visual search path | Business value of the feature |
| **Click-Through Rate (CTR)** | Clicks / Impressions | Proxy for result relevance |
| **Search Session Satisfaction** | Sessions without back-navigation or reformulation | User found what they wanted |
| **Search Abandonment Rate** | Sessions that end with no click | Bad results drive users away |

---

## ML Objectives (What the Model Optimizes)

The business metric is too coarse to train on directly. We need ML-level objectives.

### Option 1: Embedding Similarity (Metric Learning)
- **What:** Learn an embedding space where visually similar items are close (small cosine distance)
- **Training signal:** Triplets — (query, positive match, negative mismatch)
- **Pros:** Generalizes to unseen products, no category label needed
- **Cons:** Requires careful triplet mining; "similar" is subjective

### Option 2: Classification-then-Retrieval
- **What:** Train a category classifier; use category + attribute features for retrieval
- **Training signal:** Category labels, attribute tags
- **Pros:** Interpretable; easier to debug; works well for structured catalogs
- **Cons:** Can't generalize to out-of-taxonomy searches; brittle for fashion (subjective style)

### Option 3: Multi-modal Alignment (CLIP-style)
- **What:** Learn a joint image-text embedding space; image and its product title/description are "close"
- **Training signal:** (image, text description) pairs — naturally available in product catalogs
- **Pros:** Enables both image-to-image and text-to-image search; strong zero-shot performance
- **Cons:** Requires large compute; domain shift from web-pretrained CLIP to product images

**Industry choice:** Most modern systems (Pinterest, Amazon) use **metric learning + CLIP fine-tuning** in combination. Start with Option 1 for the core retrieval, add Option 3 for multi-modal queries.

---

## Constraints

These are non-negotiable requirements that shape every architectural decision:

| Constraint | Target | Why |
|---|---|---|
| **Search latency (P99)** | < 200ms end-to-end | Users abandon at >300ms |
| **Catalog size** | 10M–100M products | ANN search, not brute force |
| **Freshness** | New products indexed within 1 hour | New inventory must be searchable |
| **Availability** | 99.9% uptime | Revenue-critical feature |
| **Throughput** | 10K QPS peak | Holiday traffic spikes |

---

## ML Objective Summary (What to Optimize)

```
Primary:   Recall@10 ≥ 0.80 on human-labeled similar pairs
Secondary: MRR (Mean Reciprocal Rank) — first relevant item should appear early
Guard:     P99 latency < 200ms (system constraint, not just ML)
```

The model is **not** trained directly on conversion — conversion is too sparse and delayed. Instead we use retrieval quality (recall, MRR) as a proxy, then validate with online A/B tests measuring CTR and conversion.

---

## Interview Checkpoint

> These are questions you should be ready to answer (and ask) in an interview:

1. **"How do you define 'visual similarity' for products?"**
   - Same category? Same style? Same color? This is ambiguous — the answer shapes your training data design.

2. **"What is the north-star metric and how does ML improvement translate to business value?"**
   - Recall@10 improvement → more relevant results → higher CTR → higher conversion.

3. **"Why not train directly on conversion?"**
   - Sparse signal (most searches don't convert), delayed feedback (multi-day attribution window), confounded by price and shipping.

4. **"How do you handle the tension between precision and recall?"**
   - E-commerce prefers precision (user wants the exact item), but recall matters for broad style searches. You tune this at re-ranking time, not embedding time.
