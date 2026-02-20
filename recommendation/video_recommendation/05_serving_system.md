# Serving System Architecture

How the video recommendation system serves real-time requests at scale, orchestrating all four models (Two-Tower retrieval, Content-Based retrieval, Multi-Task ranking, Re-Ranking) to produce personalized recommendations in under 200ms.

---

## 1. System Diagram

```
                                    ┌─────────────────────────────────────────────────────┐
                                    │                  OFFLINE PIPELINE                     │
                                    │                                                       │
                                    │  ┌──────────┐    ┌──────────────┐   ┌─────────────┐  │
                                    │  │ Training  │    │  Embedding   │   │  Feature     │  │
                                    │  │ Pipeline  │    │  Pipeline    │   │  Pipeline    │  │
                                    │  │ (Airflow) │    │  (Spark)     │   │  (Spark/     │  │
                                    │  │           │    │              │   │   Flink)     │  │
                                    │  └─────┬─────┘    └──────┬───────┘   └──────┬──────┘  │
                                    │        │                 │                  │          │
                                    │        ▼                 ▼                  ▼          │
                                    │  ┌──────────┐    ┌──────────────┐   ┌─────────────┐  │
                                    │  │  Model    │    │  Embedding   │   │  Feature     │  │
                                    │  │  Registry │    │  Index       │   │  Store       │  │
                                    │  │ (MLflow)  │    │  (FAISS/     │   │ (Feast/      │  │
                                    │  │           │    │   ScaNN)     │   │  Tecton)     │  │
                                    │  └─────┬─────┘    └──────┬───────┘   └──────┬──────┘  │
                                    │        │                 │                  │          │
                                    └────────┼─────────────────┼──────────────────┼──────────┘
                                             │                 │                  │
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┼─ ─ ─ ─ ─ ─ ─ ─┼─ ─ ─ ─ ─ ─ ─ ─ ┼─ ─ ─ ─ ─
                                             │                 │                  │
                                    ┌────────┼─────────────────┼──────────────────┼──────────┐
                                    │        ▼                 ▼                  ▼          │
  ┌────────┐   ┌──────────┐        │  ┌──────────┐    ┌──────────────┐   ┌─────────────┐  │
  │ Client │──▶│ API      │───────▶│  │  Model    │    │  ANN         │   │  Feature     │  │
  │ (App)  │   │ Gateway  │        │  │  Serving  │    │  Service     │   │  Serving     │  │
  └────────┘   │ (Envoy)  │        │  │ (TF Serv/ │    │  (FAISS      │   │  (Redis/     │  │
               └──────────┘        │  │  Triton)  │    │   Server)    │   │  DynamoDB)   │  │
                    │              │  └──────────┘    └──────────────┘   └─────────────┘  │
                    │              │        ▲                 ▲                  ▲          │
                    ▼              │        │                 │                  │          │
  ┌──────────────────────────┐    │  ┌─────┴─────────────────┴──────────────────┴──────┐  │
  │   Recommendation         │    │  │                                                  │  │
  │   Orchestrator           │────│──│    Internal service calls (gRPC / HTTP2)         │  │
  │   Service                │    │  │                                                  │  │
  │   (controls the flow)    │    │  └──────────────────────────────────────────────────┘  │
  └──────────────────────────┘    │                     ONLINE SERVING                     │
                                    └──────────────────────────────────────────────────────┘
```

### Detailed Request Flow

```
User opens app (t=0ms)
│
▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ API Gateway (Envoy/NGINX)                                           t=2ms   │
│  • Rate limiting, auth, routing                                             │
│  • A/B test assignment (experiment_id → user cookie)                        │
└────────────────────────┬─────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Recommendation Orchestrator Service                                  t=5ms  │
│                                                                             │
│  1. Parse request (user_id, device, context)                                │
│  2. Fetch user profile + features (async)                                   │
│  3. Fan out to retrieval sources (parallel)                                 │
│  4. Merge & deduplicate candidates                                          │
│  5. Enrich candidates with features                                         │
│  6. Call ranking model                                                      │
│  7. Apply re-ranking                                                        │
│  8. Return final list                                                       │
│  9. Log impression for training                                             │
└────────────────────────┬─────────────────────────────────────────────────────┘
                         │
     ┌───────────────────┼───────────────────┬─────────────────┐
     │                   │                   │                 │
     ▼                   ▼                   ▼                 ▼
┌─────────┐      ┌──────────┐       ┌──────────┐      ┌──────────┐
│Feature   │      │Two-Tower │       │Content   │      │Popularity│  t=5-50ms
│Serving   │      │ANN       │       │ANN       │      │Cache     │  (parallel)
│(Redis)   │      │Retrieval │       │Retrieval │      │(Redis)   │
│          │      │(FAISS)   │       │(FAISS)   │      │          │
│• user    │      │          │       │          │      │• top-200 │
│  profile │      │• encode  │       │• build   │      │  global  │
│• video   │      │  user →  │       │  user    │      │• top-50  │
│  stats   │      │  emb     │       │  content │      │  per     │
│• real-   │      │• ANN     │       │  profile │      │  region  │
│  time    │      │  search  │       │• ANN     │      │          │
│  signals │      │  → 100   │       │  search  │      │          │
│          │      │  cands   │       │  → 100   │      │          │
└─────┬────┘      └────┬─────┘       └────┬─────┘      └────┬─────┘
      │                │                   │                 │
      │                └───────────────────┴─────────────────┘
      │                            │
      │                   ~300 unique candidates (merged & deduped)
      │                            │
      │                            ▼
      │              ┌──────────────────────────┐
      └─────────────▶│ Feature Enrichment        │                     t=50-80ms
                     │                            │
                     │ For each of 300 candidates:│
                     │ • video features           │
                     │ • engagement stats          │
                     │ • cross features            │
                     │   (user×video)              │
                     └──────────┬─────────────────┘
                                │
                                ▼
                     ┌──────────────────────────┐
                     │ Ranking Model Server      │                     t=80-140ms
                     │ (TF Serving / Triton)     │
                     │                            │
                     │ Batch inference:            │
                     │ 300 candidates × MMoE      │
                     │ → 5 predictions each       │
                     │ → combined ranking score    │
                     │                            │
                     │ GPU: ~20ms for 300 items    │
                     │ CPU: ~50ms for 300 items    │
                     └──────────┬─────────────────┘
                                │
                                ▼
                     ┌──────────────────────────┐
                     │ Re-Ranking Module          │                    t=140-160ms
                     │ (in Orchestrator process)  │
                     │                            │
                     │ • Hard filters              │
                     │ • Score adjustments          │
                     │ • MMR diversity              │
                     │ • Creator caps               │
                     │ • Exploration slots          │
                     │                            │
                     │ 300 → 30 final recs          │
                     └──────────┬─────────────────┘
                                │
                                ▼
                     ┌──────────────────────────┐
                     │ Response + Impression Log  │                    t=160-170ms
                     │                            │
                     │ • Return top-30 to client   │
                     │ • Async: log impression to  │
                     │   Kafka for training data    │
                     │   collection                 │
                     └─────────────────────────────┘
```

---

## 2. Key Components

### 2.1 API Gateway

| Aspect | Detail |
|--------|--------|
| **Technology** | Envoy, NGINX, or AWS API Gateway |
| **Responsibilities** | TLS termination, rate limiting, auth, request routing |
| **A/B test assignment** | Hash(user_id + experiment_id) → variant bucket. Sticky assignment ensures consistent experience |
| **Failover** | If recommendation service is down, route to fallback (popularity-based) |

### 2.2 Recommendation Orchestrator

The **central coordinator** — owns the request lifecycle and calls all downstream services.

```python
class RecommendationOrchestrator:
    """
    Orchestrates the full recommendation pipeline for a single request.
    Runs in a stateless microservice (e.g., Go, Java, or Python + asyncio).
    """

    async def get_recommendations(self, request: RecommendRequest) -> list[Video]:
        # Step 1: Fetch user context (async, non-blocking)
        user_context = await self.feature_service.get_user_context(request.user_id)

        # Step 2: Fan out to retrieval sources IN PARALLEL
        retrieval_tasks = [
            self.two_tower_retrieval(user_context),       # ~100 candidates
            self.content_based_retrieval(user_context),    # ~100 candidates
            self.popularity_retrieval(user_context),       # ~50 candidates
            self.subscription_retrieval(user_context),     # ~50 candidates
        ]
        results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)

        # Step 3: Merge and deduplicate
        candidates = self.merge_and_dedup(results)  # ~300 unique

        # Step 4: Enrich with features (batch lookup)
        enriched = await self.feature_service.enrich_candidates(
            candidates, user_context
        )

        # Step 5: Ranking model (batch inference)
        scored = await self.ranking_service.score_batch(enriched)

        # Step 6: Re-ranking (runs in-process, fast)
        final = self.reranker.rerank(scored, user_context)

        # Step 7: Async log impression (fire-and-forget)
        asyncio.create_task(self.log_impression(request, final))

        return final
```

**Key design decisions:**
- **Stateless**: No in-memory state, horizontal scaling via replicas
- **Async I/O**: Non-blocking calls to all downstream services
- **Parallel retrieval**: All sources fetched simultaneously → latency = max(sources), not sum
- **Timeout + fallback**: Each retrieval source has a 30ms timeout; if it fails, proceed with remaining sources

### 2.3 Feature Serving

The performance-critical data layer. Every recommendation request requires looking up hundreds of features for hundreds of candidates.

```
Feature Serving Architecture:

┌────────────────────────────────────────────────────────┐
│                    Feature Store                        │
│                                                        │
│  ┌──────────────────┐    ┌──────────────────────────┐  │
│  │  Online Store     │    │  Offline Store            │  │
│  │  (Redis Cluster)  │    │  (BigQuery / Hive)        │  │
│  │                    │    │                            │  │
│  │  • User profiles   │    │  • Training features       │  │
│  │  • Video stats     │    │  • Historical aggregates   │  │
│  │  • Real-time       │    │  • Feature backfills       │  │
│  │    counters        │    │                            │  │
│  │  • Pre-computed    │    │                            │  │
│  │    cross features  │    │                            │  │
│  │                    │    │                            │  │
│  │  Latency: <5ms    │    │  Latency: seconds          │  │
│  └──────────────────┘    └──────────────────────────┘  │
│                                                        │
│  ┌──────────────────┐    ┌──────────────────────────┐  │
│  │  Streaming Update │    │  Batch Update             │  │
│  │  (Flink / Kafka   │    │  (Spark daily job)        │  │
│  │   Streams)        │    │                            │  │
│  │                    │    │  • Video engagement stats  │  │
│  │  • Last N watched  │    │  • User category affinity  │  │
│  │  • Session count   │    │  • Creator authority score │  │
│  │  • Recent clicks   │    │                            │  │
│  └──────────────────┘    └──────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

**Feature freshness tiers:**

| Tier | Update Frequency | Examples | Store |
|------|-----------------|----------|-------|
| **Real-time** | Seconds | Last video watched, session length, click stream | Redis + Flink |
| **Near-real-time** | Minutes | Video trending score, creator upload count today | Redis + Kafka consumer |
| **Batch** | Hours/Daily | User category affinity, video lifetime stats, creator authority | Redis (loaded from Spark) |
| **Static** | Rarely | User demographics, video metadata, category taxonomy | Redis / DynamoDB |

### 2.4 ANN (Approximate Nearest Neighbor) Service

Handles the Two-Tower and Content-Based retrieval.

```
ANN Service Architecture:

┌─────────────────────────────────────────────────────┐
│  ANN Index Service                                   │
│                                                      │
│  ┌────────────────────┐  ┌────────────────────────┐ │
│  │  Two-Tower Index    │  │  Content-Based Index    │ │
│  │  (FAISS IVF-PQ)    │  │  (FAISS IVF-Flat)       │ │
│  │                      │  │                          │ │
│  │  10M video embs     │  │  10M video embs          │ │
│  │  128-d each          │  │  128-d each              │ │
│  │  Memory: ~5 GB       │  │  Memory: ~5 GB           │ │
│  │                      │  │                          │ │
│  │  Query: <5ms         │  │  Query: <5ms             │ │
│  └────────────────────┘  └────────────────────────┘ │
│                                                      │
│  Index refresh: every 4-6 hours (full rebuild)       │
│  New videos: added to a small "fresh" index hourly   │
└─────────────────────────────────────────────────────┘
```

**Index type comparison (FAISS):**

| Index Type | Memory | Query Speed | Recall@100 | Best For |
|-----------|--------|-------------|------------|----------|
| **Flat (brute-force)** | 5 GB (10M × 128 × 4B) | ~50ms | 100% (exact) | <1M items, testing |
| **IVF-Flat** | 5 GB + centroids | ~5ms | 95-98% | 1M-50M items |
| **IVF-PQ** | ~1 GB (compressed) | ~2ms | 90-95% | >50M items, memory-constrained |
| **HNSW** | ~8 GB (graph overhead) | ~1ms | 97-99% | Best speed-recall trade-off |
| **ScaNN** (Google) | ~2 GB | ~1ms | 95-98% | Google-stack, anisotropic quantization |

**Recommendation**: HNSW for best quality/speed, IVF-PQ if memory-constrained at scale.

### 2.5 Model Serving

Hosts the Multi-Task Ranking model (MMoE) for batch inference.

```
Model Serving Architecture:

┌──────────────────────────────────────────────────────┐
│  Model Serving Cluster                                │
│                                                       │
│  ┌─────────────────┐    ┌─────────────────────────┐  │
│  │  NVIDIA Triton   │    │  Model Registry          │  │
│  │  Inference Server │    │  (MLflow / SageMaker)    │  │
│  │                    │    │                           │  │
│  │  • GPU instances   │    │  • Versioned models       │  │
│  │  • Dynamic         │    │  • Canary deployment      │  │
│  │    batching         │    │  • Rollback support       │  │
│  │  • Model warmup    │    │  • A/B model routing      │  │
│  │  • Health checks   │    │                           │  │
│  └─────────────────┘    └─────────────────────────┘  │
│                                                       │
│  Inference modes:                                     │
│  • GPU (T4/A10): 300 items in ~15-20ms (preferred)    │
│  • CPU (c5.4xl): 300 items in ~40-60ms (fallback)     │
│                                                       │
│  Scaling: auto-scale on p99 latency and queue depth   │
└──────────────────────────────────────────────────────┘
```

**Model serving options comparison:**

| Platform | Pros | Cons | Best For |
|----------|------|------|----------|
| **TF Serving** | Native TF, stable, Google-proven | TF models only | TensorFlow shops |
| **NVIDIA Triton** | Multi-framework, dynamic batching, GPU-optimized | Operational complexity | Multi-model GPU serving |
| **TorchServe** | Native PyTorch, simple setup | Less mature batching | PyTorch shops |
| **Custom (FastAPI + ONNX)** | Full control, portable | Must build batching yourself | Small teams, prototyping |
| **SageMaker Endpoints** | Managed, auto-scaling | Vendor lock-in, latency overhead | AWS-native teams |

**Recommendation**: Triton for GPU-heavy workloads, TF Serving for TF teams, Custom ONNX for simplicity.

### 2.6 Impression Logger

Every recommendation served is logged for training data and debugging.

```
Impression Logging:

  Orchestrator ──async──▶ Kafka topic: "impressions"
                               │
                   ┌───────────┼───────────┐
                   ▼           ▼           ▼
             Flink consumer  Spark job   Debug DB
             (real-time      (batch      (Elasticsearch)
              feature         training
              update)         data)
```

**What gets logged per impression:**

```json
{
  "impression_id": "imp_88291002",
  "user_id": "u_382910",
  "timestamp": 1700000000,
  "experiment_id": "exp_ranking_v3",
  "variant": "treatment",
  "candidates_retrieved": 312,
  "candidates_ranked": 300,
  "final_list_size": 30,
  "recommendations": [
    {
      "video_id": "v_48291",
      "position": 1,
      "ranking_score": 7.32,
      "retrieval_source": "two_tower",
      "predictions": {"click": 0.85, "watch_time": 5.9, "like": 0.22}
    }
  ],
  "latency_ms": {
    "total": 168,
    "retrieval": 45,
    "feature_enrichment": 32,
    "ranking": 58,
    "reranking": 12,
    "overhead": 21
  }
}
```

---

## 3. Latency Budget Breakdown

Total end-to-end target: **< 200ms (p99)**

```
Component                     p50      p99      Budget    Notes
─────────────────────────────────────────────────────────────────
API Gateway                   2ms      5ms      5ms       TLS + routing
User Context Fetch            3ms      8ms      10ms      Redis lookup
                                                          (parallel with retrieval)
Retrieval (parallel):
  ├─ Two-Tower ANN            4ms      12ms     15ms      User encode + FAISS
  ├─ Content-Based ANN        3ms      10ms     15ms      Profile lookup + FAISS
  ├─ Popularity Cache         1ms      3ms      5ms       Redis sorted set
  └─ Subscription             2ms      6ms      10ms      Redis + filter
Merge & Dedup                 1ms      2ms      3ms       In-memory hash set
Feature Enrichment            15ms     35ms     40ms      Batch Redis MGET
  (300 candidates × ~10 keys)                              for 300 items
Ranking Model                 18ms     45ms     50ms      GPU batch inference
Re-Ranking                    5ms      12ms     15ms      In-process computation
Response Serialization        1ms      3ms      5ms       Protobuf
─────────────────────────────────────────────────────────────────
TOTAL                        ~55ms    ~140ms   <200ms
Impression Log (async)        —        —        N/A       Fire-and-forget
```

**Key insight**: Retrieval sources run **in parallel**, so the retrieval latency is `max(sources)`, not `sum(sources)`. This is the single most important optimization in the serving path.

---

## 4. Scaling Architecture

### Horizontal Scaling

```
                        ┌──────────────────┐
                        │   Load Balancer   │
                        └────────┬─────────┘
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
             ┌───────────┐┌───────────┐┌───────────┐
             │Orchestrator││Orchestrator││Orchestrator│  ← Stateless, scale by replicas
             │  Pod 1     ││  Pod 2     ││  Pod 3     │
             └───────────┘└───────────┘└───────────┘
                    │            │            │
         ┌─────────┴─────┬──────┴──────┬─────┴─────────┐
         ▼               ▼             ▼               ▼
  ┌─────────────┐ ┌─────────────┐ ┌──────────┐ ┌──────────────┐
  │ ANN Service  │ │ Feature     │ │ Ranking  │ │ Popularity   │
  │ (sharded by  │ │ Store       │ │ Model    │ │ Cache        │
  │  index       │ │ (Redis      │ │ (GPU     │ │ (Redis       │
  │  partition)  │ │  Cluster)   │ │  pool)   │ │  replicas)   │
  └─────────────┘ └─────────────┘ └──────────┘ └──────────────┘
```

### Scaling Strategy Per Component

| Component | Scaling Method | Bottleneck | Target |
|-----------|---------------|------------|--------|
| **Orchestrator** | Horizontal (add pods) | CPU-bound (merge, rerank logic) | <200ms p99 |
| **ANN Service** | Shard index + replicate shards | Memory (index size) | <15ms p99 |
| **Feature Store (Redis)** | Redis Cluster (hash-slot sharding) | Memory + network | <10ms p99 |
| **Ranking Model** | GPU pool + auto-scale on queue depth | GPU compute | <50ms p99 |
| **Popularity Cache** | Read replicas | Read throughput | <5ms p99 |

### Traffic Estimates

| Metric | Value | Derived |
|--------|-------|---------|
| DAU | 100M | — |
| Avg requests / user / day | 20 | Homepage + scroll |
| Peak QPS | ~50,000 | 2B requests/day, 2x peak factor |
| Candidates / request | 300 | 5 retrieval sources merged |
| Ranking inferences / second | 15M | 50K QPS × 300 candidates |
| Feature lookups / second | ~100M | 50K QPS × ~2000 feature reads |

---

## 5. Key Trade-Offs

### 5.1 Precomputation vs. Real-Time Computation

| Aspect | Precompute (Offline) | Compute at Serving Time |
|--------|---------------------|------------------------|
| **Item embeddings** | Precompute all 10M item embeddings every 4-6h | Compute per candidate at request time |
| **User embeddings** | ❌ Stale quickly (interests shift within session) | ✅ Compute from fresh features at request time |
| **Video stats** | Batch update daily | Stream updates via Flink (near-real-time) |
| **Cross features** | Infeasible (N_users × N_items combinations) | Must compute at serving time |
| **Trade-off** | Lower latency, stale data | Higher latency, fresh data |

**Rule of thumb**: Precompute what changes slowly (items), compute what changes fast (users, cross features).

### 5.2 Accuracy vs. Latency

| Decision | Accuracy Bias | Latency Bias |
|----------|--------------|--------------|
| Candidate count | More candidates (500) → better recall | Fewer candidates (200) → faster ranking |
| Ranking model size | Larger model → better predictions | Smaller / distilled model → faster inference |
| Feature count | More features (1000+) → richer signal | Fewer features (200) → faster lookup + inference |
| ANN recall | Higher recall (HNSW) → more relevant candidates | Lower recall (IVF-PQ) → faster search |

**YouTube's approach**: Use a lightweight model to score all ~1000 candidates, then a heavier model to re-score the top 100.

### 5.3 Freshness vs. Stability

| Signal | Freshness Need | Risk of Staleness |
|--------|---------------|-------------------|
| User's last watched video | Critical (minutes) | Immediate repeat recs |
| Video trending score | Important (hours) | Miss emerging trends |
| Video lifetime engagement stats | Low (daily ok) | Negligible |
| Model weights | Medium (retrain daily) | Gradual drift from user behavior changes |
| ANN index | Medium (rebuild every 4-6h) | New videos not retrievable |

### 5.4 GPU vs. CPU for Model Serving

| Aspect | GPU (T4/A10) | CPU (c5.4xlarge) |
|--------|-------------|------------------|
| **Latency** | 15-20ms / 300 candidates | 40-60ms / 300 candidates |
| **Cost** | ~$0.50/hr (T4) | ~$0.68/hr (c5.4xl) |
| **Throughput** | ~2000 batches/sec | ~200 batches/sec |
| **Scaling** | Fewer machines, harder to get GPU capacity | More machines, easy to provision |
| **Operational** | GPU drivers, CUDA versioning complexity | Standard ops |
| **When to use** | High QPS, tight latency budget | Low QPS, cost-sensitive, or GPU unavailable |

**Recommendation**: GPU for ranking (high throughput, tight latency), CPU for everything else.

---

## 6. Failure Modes and Resilience

### Degradation Strategy

```
Full system healthy
    │
    │ ANN service fails
    ▼
Fallback: skip failed retrieval source, use remaining sources
(fewer candidates, still personalized)
    │
    │ Ranking model fails
    ▼
Fallback: use retrieval scores directly (bypass ranking)
(lower quality but still relevant)
    │
    │ Feature store fails
    ▼
Fallback: use default/zero features for missing values
(degraded predictions but system stays up)
    │
    │ All personalization fails
    ▼
Fallback: serve from popularity cache
(not personalized, but users still see content)
    │
    │ Popularity cache fails
    ▼
Fallback: return hardcoded editorial picks
(absolute last resort)
```

### Specific Failure Handling

| Failure | Detection | Mitigation | Recovery |
|---------|-----------|------------|----------|
| ANN index corrupt | Health check query returns wrong results | Serve from replica, rebuild index | Auto-rebuild from last known good embeddings |
| Model serving OOM | Container restart, p99 spike | Auto-scale, circuit breaker, CPU fallback | Rolling restart with increased memory |
| Redis feature store down | Connection timeout, error rate spike | Use cached features (stale), default values | Redis Cluster self-healing, failover to replica |
| Spike in traffic (viral event) | QPS exceeds threshold | Rate limit non-critical features, reduce candidate count | Auto-scale orchestrator pods |
| Model produces NaN/Inf | Output validation check | Fallback to previous model version | Alert, investigate, redeploy |
| New model performs poorly | Online metrics regress in A/B test | Auto-rollback if metrics degrade > threshold | Root cause analysis, retrain |

### Circuit Breaker Pattern

```python
class CircuitBreaker:
    """
    Prevents cascading failures when a downstream service is unhealthy.

    States: CLOSED (normal) → OPEN (failing, use fallback) → HALF_OPEN (testing recovery)
    """
    def __init__(self, failure_threshold=5, recovery_timeout_sec=30):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_sec
        self.state = "CLOSED"
        self.last_failure_time = 0

    async def call(self, func, fallback_func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                return await fallback_func(*args, **kwargs)

        try:
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=0.03)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except (asyncio.TimeoutError, Exception):
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            return await fallback_func(*args, **kwargs)
```

---

## 7. Model Update & Deployment Pipeline

### Continuous Training and Deployment

```
┌─────────────────────────────────────────────────────────────────┐
│                    Model Update Pipeline                         │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐ │
│  │ Training  │───▶│ Offline  │───▶│ Canary   │───▶│ Full      │ │
│  │ (Daily)   │    │ Eval     │    │ Deploy   │    │ Rollout   │ │
│  │           │    │          │    │ (5%)     │    │ (100%)    │ │
│  │ Fresh data│    │ AUC,NDCG │    │ Shadow   │    │ A/B test  │ │
│  │ → new     │    │ must not │    │ mode or  │    │ for 24h   │ │
│  │   model   │    │ regress  │    │ 5% live  │    │ then full │ │
│  └──────────┘    └──────────┘    └──────────┘    └───────────┘ │
│       │               │               │               │         │
│       │          Gate: metrics    Gate: p99       Gate: online   │
│       │          pass threshold   latency OK     metrics OK     │
│       │                                                         │
│  Auto-rollback if any gate fails                                │
└─────────────────────────────────────────────────────────────────┘
```

### Embedding Index Update

```
Item embeddings change when:
  1. Model is retrained → all embeddings change → full index rebuild
  2. New videos uploaded → incremental add to index

Strategy:
  • Full rebuild every 4-6 hours via batch job
  • Small "fresh" index for videos < 6h old, rebuilt hourly
  • At query time, search both indices and merge results
```

```
┌──────────────────┐     ┌──────────────────┐
│  Main Index       │     │  Fresh Index      │
│  (10M videos,     │     │  (50K videos,     │
│   rebuilt 4-6h)   │     │   rebuilt 1h)     │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         └───── merge at query ───┘
                     │
              top-K candidates
```

### Zero-Downtime Model Swap

```python
# Blue-Green deployment for model serving
class ModelServingManager:
    """
    Two model slots: active (serving traffic) and standby (loading new model).
    Swap is atomic — no downtime.
    """
    def __init__(self):
        self.active_model = None    # Currently serving
        self.standby_model = None   # Loading next version

    def load_new_model(self, model_path: str):
        # Load into standby slot (does not affect active traffic)
        self.standby_model = load_model(model_path)
        # Warm up: run dummy inferences to fill caches
        self.standby_model.warmup(dummy_batch)

    def swap(self):
        # Atomic pointer swap
        self.active_model, self.standby_model = self.standby_model, self.active_model

    def rollback(self):
        # Old model is still in standby slot
        self.swap()
```

---

## 8. Monitoring & Observability

### Dashboard Tiers

```
Tier 1 — Business Metrics (checked daily by leadership):
  • Total watch time / user / day
  • DAU, WAU
  • User satisfaction score (composite)

Tier 2 — System Health (real-time alerts):
  • Recommendation request p50, p95, p99 latency
  • Error rate (5xx responses)
  • Model serving latency and throughput
  • Feature store hit rate and latency
  • ANN query latency

Tier 3 — Model Quality (checked daily by ML team):
  • Online CTR, watch time, like rate (per experiment)
  • Prediction calibration (predicted vs. actual CTR)
  • Feature drift (distribution shift in key features)
  • Retrieval source contribution (% of final recs from each source)
  • Exploration slot engagement rate
```

### Key Alerts

| Alert | Threshold | Action |
|-------|-----------|--------|
| p99 latency > 300ms | 5 minutes sustained | Auto-scale, page on-call |
| Error rate > 1% | 2 minutes sustained | Circuit breaker, page on-call |
| Model serving p99 > 80ms | 10 minutes sustained | Switch to CPU fallback, investigate GPU |
| Online CTR drops > 5% vs control | 1 hour sustained | Auto-rollback model, alert ML team |
| Feature store miss rate > 10% | 5 minutes sustained | Check Redis cluster health |
| ANN recall degradation | Daily check | Rebuild index, check embedding pipeline |

### Debugging "Why Was This Video Recommended?"

Essential for trust & safety, user complaints, and model debugging.

```python
def explain_recommendation(impression_id: str) -> dict:
    """
    Reconstruct the full decision chain for a specific recommendation.
    Stored in debug DB (Elasticsearch) for 30 days.
    """
    impression = fetch_impression_log(impression_id)

    return {
        "user_id": impression["user_id"],
        "video_id": impression["video_id"],
        "retrieval_source": "two_tower",           # Where it came from
        "retrieval_rank": 23,                       # Rank in retrieval
        "retrieval_score": 0.87,
        "features_used": {                          # Key features
            "user_category_affinity_gaming": 0.30,
            "video_ctr": 0.042,
            "user_x_creator_history": 4,
        },
        "ranking_predictions": {                    # Model outputs
            "click": 0.72, "watch_time": 4.8,
            "like": 0.18, "finish": 0.45, "dislike": 0.02,
        },
        "ranking_score": 5.09,
        "ranking_position": 8,                      # Before re-ranking
        "reranking_adjustments": {
            "clickbait_penalty": 0.0,
            "quality_multiplier": 1.0,
            "freshness_boost": 1.0,
        },
        "final_position": 5,                        # After re-ranking
        "experiment": "exp_ranking_v3_treatment",
    }
```

---

## 9. Common Interview Questions & Answers

### Q1: "How do you handle a cold-start user who just signed up?"

**Answer**: Graceful degradation through multiple retrieval sources:

```
New user (no watch history):
  ├── Two-Tower retrieval: ❌ No user embedding (or use demographic-only features)
  ├── Content-Based retrieval: ❌ No watch history to build profile
  ├── Popularity retrieval: ✅ Works (global trending)
  ├── Subscription retrieval: ❌ No subscriptions yet
  └── Onboarding flow: Ask user to select interests → seed category affinity

Serving flow:
  1. First session: 80% popularity + 20% interest-seed
  2. After 5-10 interactions: Two-Tower and Content-Based start contributing
  3. After 50+ interactions: Full personalized pipeline
```

### Q2: "How do you handle a viral video that suddenly gets millions of views?"

**Answer**: Multiple mechanisms:

1. **Near-real-time trending**: Flink computes view velocity; video enters popularity retrieval within minutes
2. **ANN index gap**: Video may not be in the main FAISS index yet → "fresh index" (rebuilt hourly) catches it
3. **Feature staleness**: Video engagement stats (CTR, completion rate) are unreliable with few data points → use Bayesian smoothing (blend with category prior)
4. **Auto-scaling**: Traffic spike triggers orchestrator auto-scale

### Q3: "What's the most impactful optimization you'd make to reduce latency?"

**Answer**: In order of impact:

1. **Parallel retrieval fan-out** — Changes from sequential (150ms) to parallel (40ms). Single biggest win.
2. **Batch feature lookups** — `MGET` 300 keys in one Redis call instead of 300 individual `GET` calls.
3. **GPU batched inference** — Score 300 candidates in one forward pass, not 300 separate calls.
4. **Precompute item embeddings** — Only compute user embedding at serving time; items are precomputed.
5. **ONNX model optimization** — Convert PyTorch model to ONNX with operator fusion, 2-3x speedup.

### Q4: "How do you ensure the model in production matches what you tested offline?"

**Answer**: Training-serving skew is one of the biggest practical challenges.

| Source of Skew | Cause | Prevention |
|---------------|-------|------------|
| **Feature skew** | Feature computed differently in training (batch) vs. serving (real-time) | Use the same feature store for both; log serving-time features and compare to training features |
| **Data skew** | Training data distribution differs from live traffic | Monitor feature distributions online; use temporal train/test split |
| **Label skew** | Delayed feedback (user may like a video hours later) | Use appropriate attribution windows; update labels retroactively |
| **Model version skew** | Stale model serving while new one trains | Canary deployment, shadow scoring, automated rollout |

**Best practice**: Log the exact features used at serving time → replay them in offline evaluation → compare predictions. If they match, there's no skew.

### Q5: "The ranking model is too slow. How do you speed it up?"

**Answer**: Cascading ranking (two-phase approach used by YouTube):

```
300 candidates
      │
      ▼
┌──────────────────┐
│ Light Ranker      │  ← Small model (2-layer MLP), CPU, <5ms
│ (pre-scoring)     │  ← Uses only core features (50 features)
│ Score all 300     │  ← Purpose: quickly filter to top-100
└────────┬─────────┘
         │
    Top 100
         │
         ▼
┌──────────────────┐
│ Full Ranker       │  ← Large model (MMoE, 6 experts), GPU, <20ms
│ (detailed scoring)│  ← Uses all features (700 features)
│ Score top 100     │  ← Purpose: precise ranking for final list
└────────┬─────────┘
         │
    Top 100 scored → Re-Ranking → Final 30
```

Other optimizations:
- **Model distillation**: Train a smaller "student" model from the large "teacher"
- **ONNX + TensorRT**: Compiler optimizations, operator fusion, FP16 quantization
- **Feature selection**: Remove low-importance features (measure via permutation importance)

### Q6: "How do you A/B test a new ranking model?"

**Answer**:

```
Step 1: Offline evaluation
  • Compare AUC, NDCG on held-out test set
  • Must pass: no regression on any task metric

Step 2: Shadow mode (0% traffic)
  • New model scores alongside old model
  • Compare predictions but don't serve new model's results
  • Validates latency, error rate, prediction distributions

Step 3: Canary (1-5% traffic)
  • Randomly assign 1-5% of users to new model
  • Monitor: latency, error rate, basic engagement metrics
  • Duration: 24-48 hours

Step 4: Full A/B test (50/50 split)
  • Run for 1-2 weeks for statistical significance
  • Primary metric: total watch time per user per day
  • Guardrail metrics: DAU, dislike rate, diversity
  • Minimum detectable effect: typically 0.5% for watch time

Step 5: Full rollout or rollback
  • If metrics are positive and statistically significant → 100%
  • If any guardrail is violated → rollback
```

### Q7: "How do you handle the feedback loop problem?"

**Answer**: The model only trains on data from items it previously recommended → self-reinforcing bias.

Mitigations:
1. **Exploration slots** (5-10%): Reserve slots for random/under-exposed candidates
2. **Counterfactual logging**: Log what the model predicted for items NOT shown (for offline evaluation)
3. **Inverse propensity scoring**: Weight training examples by 1/P(shown) to correct for exposure bias
4. **Multiple retrieval sources**: Popularity and subscription sources inject candidates outside the CF feedback loop

### Q8: "Walk me through what happens when I open YouTube on my phone."

**Answer** (concise version for interviews):

```
1. App sends request: {user_id, device=mobile, timestamp, session_id}

2. API Gateway: Auth, rate limit, assign A/B experiment bucket

3. Orchestrator fetches user profile from Redis (~5ms)

4. IN PARALLEL (~40ms):
   a. Two-Tower: encode user → FAISS search → 100 candidates
   b. Content-Based: aggregate recent watch embeddings → FAISS → 100 candidates
   c. Popularity: read top-200 trending from Redis → 50 candidates
   d. Subscriptions: recent uploads from followed creators → 50 candidates

5. Merge & dedup: ~300 unique candidates

6. Feature enrichment: batch fetch video features from Redis (~30ms)

7. Ranking model: MMoE scores all 300 on GPU (~20ms)
   → P(click), E[watch_time], P(like), P(finish), P(dislike)
   → weighted combination → single score per candidate

8. Re-ranking (~10ms):
   → filter already-watched, policy violations
   → clickbait penalty, quality boost, freshness boost
   → MMR diversity, creator caps
   → insert 3 exploration candidates

9. Return top-30 to client, async-log impression to Kafka

Total: ~160ms p50, ~200ms p99

10. As user scrolls and watches, events stream back to Kafka
    → update real-time features
    → next request gets fresh context
```

---

## 10. Technology Stack Summary

| Layer | Technology Options | Recommended |
|-------|-------------------|-------------|
| **API Gateway** | Envoy, NGINX, Kong | Envoy (gRPC native) |
| **Orchestrator** | Go, Java (Spring), Python (FastAPI + asyncio) | Go (low latency, high concurrency) |
| **Feature Store** | Feast, Tecton, custom Redis | Feast + Redis (open source) |
| **ANN Index** | FAISS, ScaNN, Milvus, Pinecone | FAISS (self-hosted) or Pinecone (managed) |
| **Model Serving** | Triton, TF Serving, TorchServe, SageMaker | Triton (GPU) or TF Serving (TF models) |
| **Streaming** | Kafka, Kinesis, Pulsar | Kafka (industry standard) |
| **Stream Processing** | Flink, Kafka Streams, Spark Structured Streaming | Flink (true streaming) |
| **Batch Processing** | Spark, Dataflow | Spark (ecosystem maturity) |
| **Model Registry** | MLflow, SageMaker Model Registry, Weights & Biases | MLflow (open source) |
| **Monitoring** | Prometheus + Grafana, Datadog | Prometheus + Grafana |
| **Experiment Platform** | Custom, Optimizely, LaunchDarkly | Custom (ML-specific needs) |
| **Container Orchestration** | Kubernetes | Kubernetes (industry standard) |
