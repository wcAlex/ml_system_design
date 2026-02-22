# 11 — Summary

## Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        PYMK SYSTEM ARCHITECTURE                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────── DATA SOURCES ───────────────────────────────┐
│                                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │  Graph DB   │  │ Profile DB  │  │ Interaction  │  │  Label Store    │  │
│  │             │  │             │  │    Logs      │  │                 │  │
│  │ 500B edges  │  │ 1B profiles │  │ Clicks/Views │  │ Accept/Dismiss  │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘  └────────┬────────┘  │
└─────────┼────────────────┼────────────────┼───────────────────┼────────────┘
          │                │                │                   │
          └────────────────┴────────────────┴───────────────────┘
                                      │
                                      ▼
┌─────────────────────────── OFFLINE PIPELINE ───────────────────────────────┐
│                          (Spark / distributed)                               │
│                                                                              │
│  ┌────────────────────┐    ┌──────────────────────┐    ┌─────────────────┐ │
│  │  Feature           │    │  GNN Training         │    │  Label          │ │
│  │  Engineering       │    │  (GraphSAGE/PinSage)  │    │  Generation     │ │
│  │                    │    │                       │    │                 │ │
│  │ - Graph metrics    │    │ - Mini-batch sampling │    │ - Positive:     │ │
│  │ - Profile encoding │    │ - In-batch negatives  │    │   accepted conn │ │
│  │ - Skill overlap    │    │ - 2-layer, 128-d      │    │ - Negative:     │ │
│  └────────┬───────────┘    └──────────┬────────────┘    │   14-day no-op  │ │
│           │                           │                  └─────────────────┘ │
│           │                ┌──────────▼────────────┐                        │
│           │                │  Node Embeddings       │                        │
│           │                │  (1B × 128-d)          │                        │
│           │                └──────────┬────────────┘                        │
│           │                           │                                      │
│           │                ┌──────────▼────────────┐                        │
│           │                │  ANN Index Build       │                        │
│           │                │  (FAISS HNSW / ScaNN)  │                        │
│           │                └──────────┬────────────┘                        │
│           │                           │                                      │
│           │         ┌─────────────────▼─────────────────┐                  │
│           │         │   Candidate List Generation         │                  │
│           │         │   Graph traversal + ANN union       │                  │
│           │         │   → top 1,500 per user              │                  │
│           │         └─────────────────┬─────────────────┘                  │
└───────────┼───────────────────────────┼────────────────────────────────────┘
            │                           │
            └──────────────┬────────────┘
                           │  write
                           ▼
┌──────────────────────── STORAGE LAYER ────────────────────────────────────┐
│                                                                             │
│  ┌──────────────────────────┐    ┌──────────────────────────────────────┐  │
│  │  Feature Store           │    │  ANN Index                           │  │
│  │  (Redis cluster)         │    │  (distributed FAISS, 32 shards)      │  │
│  │                          │    │                                      │  │
│  │  - node embeddings       │    │  - 1B user vectors                   │  │
│  │  - profile features      │    │  - supports ~10K QPS                 │  │
│  │  - graph features        │    │  - 128GB with int8 quant             │  │
│  │  - candidate lists       │    │                                      │  │
│  │  - behavioral signals    │    │                                      │  │
│  └──────────────┬───────────┘    └──────────────────────┬───────────────┘  │
└─────────────────┼─────────────────────────────────────────┼────────────────┘
                  │                                           │
                  └──────────────────┬────────────────────────┘
                                     │  real-time read
                                     ▼
┌──────────────────────── ONLINE SERVING ────────────────────────────────────┐
│                                                                              │
│  User Request                                                                │
│       │                                                                      │
│       ▼                                                                      │
│  ┌──────────────────────────────────────────────────────┐                   │
│  │  PYMK Ranking Service  (stateless, horizontally scaled) │               │
│  │                                                       │                   │
│  │  1. Check Redis cache (L1/L2)                         │                   │
│  │  2. Fetch precomputed candidates (1,500)              │                   │
│  │  3. Batch-fetch features (one pipeline call)          │                   │
│  │  4. Run ranking model (MLP, ~10ms)                    │                   │
│  │  5. Apply business rules + diversity filters          │                   │
│  │  6. Return top 20                                     │                   │
│  └──────────────────────────────────────────────────────┘                   │
│                                                                              │
│  Target latency: < 100ms P99                                                 │
│  Target throughput: 25,000 QPS peak                                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Map

| Component | Technology | Purpose |
|---|---|---|
| Graph DB | Custom adjacency list on HBase / Neptune | Store and query connection edges |
| Profile DB | DynamoDB | User profile feature retrieval |
| Interaction Log Store | Hive / BigQuery (Parquet) | Training data, label generation |
| GNN Training | PyTorch + PyG, distributed on GPU cluster | Learn node embeddings |
| Candidate Generation | Spark | Graph traversal + ANN union |
| ANN Index | FAISS HNSW (32 shards) | Billion-scale approximate nearest neighbor |
| Feature Store | Redis cluster | Low-latency feature retrieval |
| Ranking Model | XGBoost or MLP (TorchServe) | Score and rank 1,500 candidates |
| Serving Service | Python (FastAPI) | Orchestrate online pipeline |
| Cache | Redis (L1 in-process, L2 Redis) | Reduce compute for repeat requests |
| Stream Processing | Kafka + Flink | Real-time cache invalidation on new connections |

---

## Data Flow Summary

```
New connection event:
  Graph DB ← writes edge
  Kafka topic ← publishes event
  Cache invalidation service ← consumes event → Redis DEL for both users

Daily offline batch:
  Graph DB → Spark → features + embeddings → Redis + FAISS index

User opens PYMK:
  API → Redis (cache check) → candidates + features → MLP ranker → top-20
```

---

## Model Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TWO-STAGE PIPELINE                              │
│                                                                      │
│  ┌─────────────────────────┐    ┌───────────────────────────────┐   │
│  │  Stage 1: Generation    │    │  Stage 2: Ranking             │   │
│  │                         │    │                               │   │
│  │  Graph traversal:       │    │  Two-Tower MLP:               │   │
│  │  - 2nd-degree BFS       │    │  - GNN embedding (128-d)      │   │
│  │  - Filter by cn ≥ 2     │    │  - Profile features           │   │
│  │  → ~1,000 candidates    │    │  - Behavioral features        │   │
│  │                         │    │  - Graph heuristic features   │   │
│  │  ANN Search:            │    │  → score per candidate        │   │
│  │  - Query embedding      │    │                               │   │
│  │  - HNSW top-500         │    │  Business rules:              │   │
│  │                         │    │  - Dedup, block, dismiss      │   │
│  │  Union → 1,500 total    │    │  - Diversity enforcement      │   │
│  └──────────┬──────────────┘    └───────────────┬───────────────┘   │
│             │                                   │                    │
│             └──────────────────────────────────►│ → top 20          │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions Recap

| Decision | Choice | Reasoning |
|---|---|---|
| GNN architecture | GraphSAGE (inductive) | Handles new users; scalable with neighbor sampling |
| Training loss | In-batch negatives (sampled softmax) | Efficient, harder negatives, scales with batch size |
| Candidate generation | Hybrid (graph + ANN) | Graph catches structural candidates; ANN catches semantic |
| Ranking model | Two-tower MLP (not GNN) | Precomputable embeddings → fast ANN at serving time |
| Offline vs. online | Heavy work offline, lightweight online | Only way to meet < 150ms at 1B scale |
| GNN layers | 2 layers | Beyond 2 causes over-smoothing; 2-hop is sufficient for social graphs |
| Negative sampling | 14-day impression window | Cleaner negatives than random; captures explicit non-interest |
