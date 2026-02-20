# 10 — Challenges & Talking Points

## The "What Could Go Wrong?" Section

Interviewers love asking about failure modes. Knowing the challenges, why they happen, and how to mitigate them demonstrates senior-level thinking. This is often what separates a good answer from a great one.

---

## Challenge 1: Query Image vs. Catalog Image Distribution Shift

### What It Is
Users upload real-world photos (taken in bad lighting, messy backgrounds, weird angles). Your catalog has clean studio shots. The model trained on catalog images may produce poor embeddings for messy query photos.

### Example
User uploads a dark, blurry photo of a sneaker from their bedroom floor. The embedding lands far from the catalog embedding of that sneaker (which has a perfect white background, multiple angles, professional lighting).

### Mitigations
| Approach | Description | Trade-off |
|---|---|---|
| Augmentation | Simulate real-world conditions (blur, noise, dark) during training | May hurt performance on clean queries |
| Query-specific encoder | Separate model head for noisy queries | More complex, two encoders to maintain |
| Image quality gating | Reject/warn on low-quality images | Bad UX if threshold is too aggressive |
| Ensemble | Average CLIP embedding + DINOv2 embedding | Better robustness; double latency |

### Interview Angle
> "How does your model handle distribution shift between training data and real user queries?"

---

## Challenge 2: Multi-Object Query Images

### What It Is
User uploads a photo of a full outfit — dress, shoes, bag, jewelry. What are they searching for?

### Options
1. **Return results for the whole image** (simple) — often confusing, low precision
2. **Object detection → user selects region** (Pinterest Lens approach) — best UX, adds latency + infra
3. **Return results per detected object** — clutters UI
4. **Infer intent from the most prominent object** — auto-select largest bounding box

### Interview Angle
> "What happens when the query image contains multiple products?"

---

## Challenge 3: Catalog Freshness vs. Index Consistency

### What It Is
New products are added continuously (thousands per hour on a large platform). The ANN index is rebuilt nightly. New products are invisible for up to 24 hours.

### Mitigations
| Strategy | Freshness | Complexity |
|---|---|---|
| Full nightly rebuild | 24 hours | Low |
| Incremental hourly append | 1 hour | Medium |
| Real-time streaming (Kafka + Milvus) | Minutes | High |
| Hybrid: rebuild + priority queue | Configurable | Medium |

**Key insight:** Appending to FAISS indexes degrades performance (no re-clustering). You eventually need a full rebuild. Use Milvus or Qdrant if you need real-time freshness.

---

## Challenge 4: Long-Tail Products

### What It Is
Your catalog has millions of SKUs, but 80% of searches are for the top 20% of popular items. For niche, artisan, or brand-new products with few views or interactions, the model may learn poor embeddings.

### Causes
- Little training signal (few click/purchase pairs)
- No user behavior data to mine hard negatives from
- May be visually unique — no similar products in catalog to anchor the embedding

### Mitigations
- Increase sampling weight for rare products during training
- Use text embedding (product description) as auxiliary supervision
- Rely on attribute matching and category filtering for tail products
- Flag: "limited visual matches — showing similar styles" in the UI

---

## Challenge 5: Semantic vs. Visual Similarity Gap

### What It Is
Users often want **semantic** matches, not just visual matches.

**Example:** User uploads photo of a vintage-style leather sofa. The model returns products that are visually similar (same color, similar shape) but may be cheap imitations. The user actually wants something with "vintage aesthetic" — a semantic concept.

**Another example:** User uploads a photo of a product in red and wants to find "the same in blue" — visual similarity is now an anti-goal.

### Mitigations
- Multi-modal search: use text refinement ("same style but in blue") to steer the query embedding
- Attribute-based filtering: extract color attribute, allow user to change it post-retrieval
- User controls: "show exact matches" vs. "show similar style" toggle

---

## Challenge 6: False Negatives in Training Data

### What It Is
Two products are labeled as "negative pair" (different SKUs) but are actually visually identical (e.g., same manufacturer, different resellers). Pushing these apart during training hurts embedding quality.

### Detection
```python
def find_potential_false_negatives(embeddings, product_ids, threshold=0.95):
    """Find pairs with very high similarity that are labeled as different products."""
    sim_matrix = embeddings @ embeddings.T
    pairs = []
    for i in range(len(product_ids)):
        for j in range(i+1, len(product_ids)):
            if sim_matrix[i,j] > threshold and product_ids[i] != product_ids[j]:
                pairs.append((product_ids[i], product_ids[j], sim_matrix[i,j]))
    return pairs
```

### Mitigation
- Deduplicate catalog by perceptual hash (pHash) before training
- Use confident learning (Cleanlab) to identify noisy training labels
- Apply softer negative penalties (e.g., soft margin or label smoothing)

---

## Challenge 7: Model Serving Latency Regression

### What It Is
A newer, more accurate model (e.g., ViT-L vs. ViT-B) improves offline metrics but increases serving latency beyond the P99 budget.

### Trade-off Matrix
| Model | Recall@10 | GPU Latency | Memory |
|---|---|---|---|
| CLIP ViT-B/32 | 78% | 10ms | 350MB |
| CLIP ViT-B/16 | 84% | 20ms | 350MB |
| CLIP ViT-L/14 | 88% | 45ms | 900MB |

If your latency budget is 40ms for the embedding step, ViT-L/14 doesn't fit. Options:
- Knowledge distillation: train ViT-B student to mimic ViT-L teacher
- Quantization: INT8 ViT-L ≈ latency of FP32 ViT-B
- Separate fast/slow lanes: use ViT-B for all queries, ViT-L only for high-value users

---

## Challenge 8: Popularity and Exposure Bias

### What It Is
Products that appear higher in search results get more clicks → more training data → better embeddings → even higher ranking. **The rich get richer.**

New products or niche items never accumulate enough signal to compete.

### Mitigations
- **Exploration traffic:** Reserve 5% of traffic for exploration (randomly order results) — generates diverse training signal
- **Debiased training:** Weight training examples inversely by item popularity
- **Inverse propensity scoring (IPS):** Correct for exposure probability in click-through training

---

## Challenge 9: Index Rebuild on Model Retrain

### What It Is
When the embedding model is updated (new training run), ALL product embeddings are stale — they were generated by the old model. You must re-embed the entire catalog and rebuild the index.

**Cost:** For 10M products, re-embedding takes hours on GPU clusters.

### Mitigations
- Continuous background re-embedding job (runs constantly, catches up in hours)
- Versioned indexes: serve old index during rebuild, hot-swap atomically
- Incremental fine-tuning (small updates don't change embedding space much) — risky

---

## Challenge 10: Category Confusion

### What It Is
An embedding model trained on all categories may conflate visually similar but categorically different items.

**Example:** A pink backpack and a pink handbag may have similar visual embeddings but are completely different products.

### Mitigation
- Add category classification loss to the training (multi-task learning)
- Apply category pre-filter before ANN search (query side: detect category of query image first)
- Re-rank with hard category match requirement

---

## Summary: Challenge Prioritization

| Challenge | Frequency | Impact | Difficulty to Fix |
|---|---|---|---|
| Distribution shift (real vs. catalog) | High | High | Medium |
| Freshness / new product indexing | High | Medium | Medium |
| Popularity bias | High | Medium | Medium |
| Multi-object query | Medium | High | High |
| Long-tail products | Medium | Low-Medium | Medium |
| False negatives in training | Medium | Medium | Medium |
| Latency regression on model upgrade | Low | High | Low |

---

## Interview Checkpoint

1. **"What's the biggest technical challenge in building an image search system?"**
   - Distribution shift + freshness. Most other things are engineering problems; these are fundamental ML challenges.

2. **"How do you prevent the system from over-favoring popular items?"**
   - Exploration traffic + IPS debiasing. This is an important fairness and diversity concern.

3. **"What happens when you ship a new model — how do you update the index?"**
   - Full re-indexing pipeline. Background job, versioned indexes, atomic hot-swap. No downtime.

4. **"How do you handle a query image where you can't detect any recognizable object?"**
   - Return a 400 or fallback to trending/popular items with a "we couldn't identify the item" message. Never return random results.
