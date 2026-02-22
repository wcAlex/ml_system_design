# 08 — Model Hosting & Inference

## Serving Architecture Overview

```
                  ┌─────────────────────────────────┐
                  │        OFFLINE PIPELINE           │
                  │  (nightly or incremental)         │
                  │                                   │
   Graph DB  ────►│  1. Graph preprocessing           │
   Profile DB────►│  2. GNN training / fine-tuning    │
   Logs ─────────►│  3. Node embedding generation     │
                  │  4. ANN index build               │
                  │  5. Feature store update          │
                  └────────────────┬────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
            ┌──────────────┐           ┌─────────────────┐
            │  ANN Index   │           │  Feature Store  │
            │  (FAISS/     │           │  (Redis /       │
            │   ScaNN)     │           │   DynamoDB)     │
            └──────┬───────┘           └────────┬────────┘
                   │                            │
                   └─────────────┬──────────────┘
                                 │
                  ┌──────────────▼──────────────────┐
                  │         ONLINE SERVING           │
                  │                                  │
  User request ──►│  1. Auth + rate limit            │
                  │  2. Fetch precomputed candidates │
                  │     (or ANN query)               │
                  │  3. Fetch features (batch)       │
                  │  4. Run ranking model inference  │
                  │  5. Apply business rules         │
                  │  6. Return top-K                 │
                  └──────────────────────────────────┘
```

---

## Component 1: ANN Index

### Purpose
Given a query user's embedding, find the top-K most similar user embeddings across all 1B users.

### Index Choice

| Index Type | Recall@100 | QPS (single machine) | Memory | Notes |
|---|---|---|---|---|
| FAISS IVF-PQ | 90–95% | 500–2000 | Low (compressed) | Good for cost-sensitive |
| FAISS HNSW | 95–99% | 200–500 | High (full vectors) | Best recall |
| ScaNN (Google) | 97–99% | 2000–5000 | Low | Best QPS/recall tradeoff |
| Approximate (Spotify Annoy) | 85–92% | 1000+ | Moderate | Simpler operationally |

**Recommended:** ScaNN or FAISS-HNSW for production.

### Sharding Strategy for 1B Users
```
1B users × 128-d × 4 bytes (float32) = 512GB raw
With int8 quantization: 128GB
With FAISS IVF-PQ (4-bit): ~32GB

Shard across machines: 32 shards × ~4GB = 1 shard per machine
ANN query: broadcast to all shards, each returns top-K local results,
           merge and return global top-K
```

### ANN Index Refresh
- Full rebuild: weekly (expensive, ~hours)
- Incremental update: for new users, compute their embedding and insert into index (HNSW supports insertion; IVF-PQ typically requires rebuild)
- New users' embeddings available within 1 hour of joining (profile-only embedding via two-tower query tower)

---

## Component 2: Feature Store

### Purpose
Provide low-latency retrieval of precomputed features for (user, candidate) pairs during ranking.

### Data Stored

| Feature Type | Storage | TTL | Access Pattern |
|---|---|---|---|
| Node embeddings (128-d) | Redis cluster | 24h | Key: user_id |
| Graph features (common neighbors, etc.) | Redis or DynamoDB | 6h | Key: (user_id, candidate_id) |
| Profile features | DynamoDB | 1h | Key: user_id |
| Behavioral features (recent views) | Redis | 1h | Key: (user_id, candidate_id) |
| Candidate lists per user | Redis | 4h | Key: user_id |

### Feature Retrieval at Serving Time
```python
# Batch fetch for all 1,000 candidates at once
user_features = feature_store.mget([query_user_id])
candidate_features = feature_store.mget(candidate_ids)  # batch of 1,000
pair_features = feature_store.mget([(query_id, c_id) for c_id in candidate_ids])
```

**Key optimization:** Use batch/pipeline requests to the feature store. Single round-trip for all 1,000 candidates instead of 1,000 individual requests. Reduces latency from ~50s to ~20ms.

---

## Component 3: Ranking Model Serving

### Model Format
- Export trained model to **ONNX** or **TorchScript**
- Deploy via **TorchServe** or **Triton Inference Server**
- Advantage: runtime optimization (fusion, quantization), language-agnostic serving

### Inference Batch Size
- Input: feature matrix of shape (num_candidates, num_features) = (1000, ~100)
- Single forward pass scores all 1,000 candidates in one batch
- Typical latency: 5–15ms on CPU; 1–3ms on GPU

### Model Size Considerations

| Model | Parameters | Inference latency (1000 candidates) | Notes |
|---|---|---|---|
| XGBoost (500 trees) | ~5MB | 1–3ms (CPU) | Very fast |
| Two-tower MLP (small) | ~2M params | 3–8ms (CPU) | Good balance |
| Two-tower MLP (large) | ~50M params | 10–20ms (CPU) | May need GPU |
| GNN ranker | ~100M params | 50–100ms | Only for re-ranking top-50 |

### Quantization
For the ranking MLP: INT8 quantization reduces model size 4× and speeds up inference 2–3× with < 1% accuracy drop. Use PyTorch dynamic quantization:
```python
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

---

## Component 4: Caching

### L1 Cache: In-Process Memory (per serving pod)
- Cache top-K results for the most recently served users
- TTL: 5 minutes
- Hit rate: ~20–30% (users often refresh the page)
- Memory: 10,000 users × 1KB per result set = ~10MB per pod

### L2 Cache: Redis (shared across pods)
- Cache precomputed rankings per user
- TTL: 1–4 hours
- Invalidate immediately when user forms a new connection
- Hit rate: ~60–70% of requests
- Memory: 1M active users × 2KB = ~2GB

### Cache Invalidation
```
Event: user A connects with user B
Action:
  1. Remove B from A's candidate list in cache
  2. Remove A from B's candidate list in cache
  3. Trigger partial refresh of A's and B's candidate sets
  4. Invalidate ranking cache entries for A and B

Mechanism: connection event → Kafka topic → cache invalidation service → Redis DEL
Latency target: < 5 seconds for cache invalidation after connection event
```

---

## Component 5: Business Rules Layer

Applied after ranking, before returning results:

```python
def apply_business_rules(ranked_candidates, query_user):
    result = []
    for candidate in ranked_candidates:
        # Hard exclusions
        if candidate.id in query_user.blocked_ids:
            continue
        if candidate.id in query_user.connection_ids:
            continue
        if candidate.id in query_user.dismissed_ids:
            continue
        if candidate.is_deactivated or candidate.is_banned:
            continue
        # Soft diversity rules
        result.append(candidate)

    # Ensure diversity: max 5 from same company, max 3 from same school
    result = enforce_diversity(result, max_per_company=5, max_per_school=3)

    return result[:limit]
```

---

## Operational Considerations

### Throughput Requirements
```
Daily active users: ~300M
PYMK requests per DAU per day: ~3 (homepage load, mobile load, refresh)
Total QPS: 300M × 3 / 86,400 = ~10,000 QPS peak (×2–3 for peak traffic)
Peak QPS: ~25,000
```

### Deployment Pattern
- Stateless ranking service: horizontally scalable, auto-scale on CPU/latency
- ANN index: stateful, requires careful shard management and rolling updates
- Feature store: Redis cluster with replica reads

### Graceful Degradation
If the full ranking pipeline is overloaded or the GNN embedding is stale:
1. Fall back to precomputed static rankings
2. Fall back to graph heuristic scores (common neighbors only)
3. Fall back to cached results from last serving, even if stale

---

## Interview Checkpoint

**Q: How do you update the ANN index as new users join?**

New users arrive at ~thousands/second at 1B scale. Options:
1. **HNSW with online insertion:** Insert new user embedding as it's computed (within 1 hour of joining). HNSW supports dynamic insertion without full rebuild.
2. **Tiered index:** Keep a small "recent users" index that is rebuilt hourly; older users stay in the main index. Merge results at query time.
3. **Cold-start fallback:** For very new users (< 1 hour old), don't query ANN at all — use graph-only candidates (they have no embedding yet). After 1 hour once they've set up their profile, generate their embedding and add to ANN index.

**Q: How do you serve 25,000 QPS with 150ms latency?**

Key techniques:
1. Precompute everything expensive offline (GNN inference, graph features)
2. Batch all feature store reads into one pipeline call
3. Use a lightweight ranker (XGBoost or small MLP) that runs in 5–15ms
4. Layer L1 + L2 caching to reduce compute for repeat requests
5. Horizontal scaling of the stateless ranking service
6. Regional deployment — serve users from geographically close data centers
