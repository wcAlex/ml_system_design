# 12 — Scaling

This document covers how the PYMK system scales across three dimensions:
1. **Graph data scale** — more users, more edges
2. **Model scale** — larger GNN, richer features
3. **Inference scale** — more QPS, lower latency

---

## Scaling Dimension 1: Graph Data

### The Challenge
- 1B nodes, 500B edges
- Growing at ~10M new users/month (~120M/year)
- Each new user brings ~100 new edges (initial connections)

### Graph Storage Sharding

**Approach: Hash-based sharding on user_id**
```
User's connections are stored on shard:
  shard_id = hash(user_id) % num_shards

For 64 shards:
  - 500B edges / 64 = ~8B edges per shard
  - Each shard: ~10–15GB of edge data
  - Fits on commodity machines
```

**Problem with hash sharding:** 2nd-degree traversal requires cross-shard lookups. User A is on shard 5; User A's neighbors are distributed across all 64 shards.

**Solution: Community-based sharding**
- Partition users by graph community (Louvain, METIS)
- Users in the same community on the same shard
- ~80% of 2nd-degree traversals happen within one shard
- Remaining ~20% require cross-shard fan-out

### Graph Processing at Scale: GraphX / DGL

For the offline pipeline (computing graph features, running GNN):

| Framework | Scale | Notes |
|---|---|---|
| NetworkX | < 1M nodes | In-memory only, Python |
| GraphX (Spark) | 10M–1B nodes | Distributed message passing on Spark |
| DGL (Deep Graph Library) | 100M+ nodes | GPU-accelerated GNN training |
| PyG (PyTorch Geometric) | Research → 100M nodes | Flexible, research-grade |
| PinSage architecture | 3B+ nodes | Custom; mini-batch sampling is the key |

**For 1B nodes:** Use DGL or a PinSage-style architecture with mini-batch subgraph sampling. The key insight: you never load the full graph into memory — you sample computation subgraphs.

---

## Scaling Dimension 2: GNN Model

### Mini-Batch Subgraph Sampling (the core technique)

Standard GCN requires the full graph in memory. Mini-batch sampling fixes this:

```
For each training batch (B seed nodes):
  1. Sample K1 neighbors per seed → B × K1 layer-1 nodes
  2. Sample K2 neighbors per layer-1 node → B × K1 × K2 layer-0 nodes
  3. Total nodes in mini-batch: B × (1 + K1 + K1*K2)

With B=256, K1=10, K2=10:
  Total nodes: 256 × (1 + 10 + 100) = 28,416
  GPU memory per batch: manageable (< 1GB)
```

This is why GraphSAGE and PinSage are production-viable at billion-node scale.

### Distributed GNN Training

**Data Parallelism (recommended):**
```
8 GPU machines, each with a copy of the model
Each GPU processes a different mini-batch
Gradients synchronized via AllReduce (NCCL)
```

**Partition + train (alternative):**
```
Split graph into P partitions (e.g., 64)
Each partition trains a GNN independently
Cross-partition node embeddings shared via parameter server
```

**Production setup (LinkedIn-scale):**
- ~100 GPU machines (8 × A100 each)
- Mini-batch size: 2048 seed nodes
- Training time for one epoch on 1B nodes: ~10–24 hours
- Full retraining: weekly
- Incremental fine-tuning (on last 7 days of data): daily

### Model Size Scaling

| Model | Params | Embedding quality | Training time | Notes |
|---|---|---|---|---|
| GCN (2-layer, 64-d) | ~500K | Baseline | Hours | Fast, simple |
| GraphSAGE (2-layer, 128-d) | ~2M | Good | ~10h | Production-ready |
| GraphSAGE (2-layer, 512-d) | ~30M | Better | ~24h | Better but slower |
| GAT (4-head, 2-layer, 256-d) | ~10M | Better on heterogeneous | ~20h | Good for typed edges |
| LLM-enhanced GNN | 768M+ | Best | Days | Too slow for daily retrain |

**Practical choice:** GraphSAGE with 128-d embeddings is the sweet spot. Increasing to 256-d gives marginal improvement; the bottleneck is usually training data quality, not model capacity.

---

## Scaling Dimension 3: Inference

### Throughput Calculation

```
1B users × 3 PYMK requests/day = 3B requests/day
3B / 86,400 seconds = ~35,000 RPS average
Peak (2× average): ~70,000 RPS

Serving budget per request: ~50ms of compute (after cache)
With 70% cache hit rate:
  Actual compute requests: 70,000 × 0.3 = ~21,000 RPS requiring full pipeline
```

### Horizontal Scaling Strategy

**Ranking Service (stateless):**
```
21,000 RPS × 50ms CPU time per request = 1,050 CPU-seconds per second
With 8 vCPU per machine: 1,050 / 8 = ~132 machines
Add 50% headroom for burst: ~200 machines for ranking service
```

**ANN Index (stateful):**
```
ANN queries = requests that miss candidate cache = 21,000 RPS × 0.3 = 6,300 RPS
1 ANN query = ~5ms on 1 shard
32 shards in parallel: 5ms total
Throughput per shard: 200 QPS
Needed: 6,300 / 200 = 32 shards (coincidentally matches our storage sharding)
```

**Feature Store (Redis):**
```
Each request fetches ~1,000 candidate features + 1 query user feature
Batch fetch = 1 pipeline call, ~20ms
Redis throughput: 100K+ operations/second per node
Needed: ~30–50 Redis nodes in a cluster
```

### Caching Impact

```
Without caching: 70,000 RPS → full pipeline
With 70% L2 cache hit rate: 21,000 RPS → full pipeline
Latency for cache hits: < 5ms (Redis GET)
Latency for cache misses: ~50–80ms

Cache size for 1% of DAU (300M × 0.01 = 3M users):
  3M × 2KB per cached recommendation = 6GB → fits in a moderate Redis cluster
```

### Embedding Serving at Scale

**Problem:** 1B users × 128-d × 4 bytes = 512GB — too large for a single Redis instance.

**Solution: Tiered embedding storage:**
```
Hot embeddings (top 1% most active users):
  - Stored in Redis (fast, ~5GB for 10M users)
  - Served in < 1ms

Cold embeddings (remaining 99%):
  - Stored in distributed KV store (e.g., RocksDB on SSD)
  - Served in 5–20ms
  - Triggered less frequently (inactive users)
```

---

## Scaling Dimension 4: Incremental Updates

### The Problem
GNN retraining from scratch takes 10–24 hours. But the graph changes continuously: ~10M new connections/day = ~115 new edges/second. Running a full retrain daily is expensive; skipping updates means stale embeddings.

### Solution: Incremental Training

**Strategy 1: Fine-tune on daily delta**
```
Every night:
  - Load yesterday's trained checkpoint
  - Fine-tune on only the last 24 hours of new connections
  - Update embeddings only for nodes whose neighborhood changed significantly
  - Run time: ~2–4 hours (vs. 10–24 hours for full retrain)
```

**Strategy 2: Neighborhood change detection**
```
For each node, track a "staleness score":
  staleness = (num_new_neighbors_last_24h) / total_degree

If staleness > threshold (e.g., 5%): recompute embedding
Otherwise: keep cached embedding

Result: only ~5–10% of nodes need daily embedding refresh
```

**Strategy 3: Streaming GNN updates (advanced)**
```
New connection A–B arrives:
  1. Update A's embedding using A's updated neighborhood
  2. Update B's embedding using B's updated neighborhood
  3. Update A's and B's neighbors' embeddings (1-hop affected)
  4. Publish new embeddings to embedding store + ANN index

Latency: embedding update within minutes of connection
Used by: Pinterest for real-time PinSage updates
```

---

## Scaling Summary Table

| Scale challenge | Solution | Complexity |
|---|---|---|
| Graph too large for memory | Mini-batch subgraph sampling | Medium |
| GNN training time | Distributed training (8× GPU machines) | Medium |
| Embedding storage (512GB) | Quantization + tiered storage | Low |
| ANN index for 1B vectors | Sharded FAISS (32 shards) | Medium |
| Ranking throughput (70K RPS) | Stateless horizontal scaling + caching | Low |
| Daily retraining cost | Incremental fine-tuning on delta | Medium |
| Real-time graph changes | Streaming embedding updates (Kafka + Flink) | High |
| Feature store latency | Batch Redis pipeline; hot/cold split | Low |

---

## Interview Checkpoint

**Q: How would you scale this system from 10M to 1B users?**

Key transitions:
1. **10M → 100M:** Single-machine ANN index becomes too slow → shard FAISS across 8 machines. Graph traversal still manageable with in-memory adjacency list per shard.
2. **100M → 1B:** Full GNN retraining daily becomes too slow → switch to incremental fine-tuning. Redis for embeddings needs tiered storage. Ranking service needs ~200 machines.
3. **At 1B:** The system as described above. The biggest challenges become operational: managing 32 ANN shards, orchestrating daily pipeline, ensuring cache invalidation at sub-second latency.

**Q: What is the biggest bottleneck in the system?**

In practice, it is usually **embedding freshness**, not compute throughput. The offline pipeline takes hours, so embeddings are always somewhat stale. The second bottleneck is **candidate generation recall** — if Stage 1 misses good candidates, Stage 2 cannot recover them. Improving Stage 1 recall (e.g., from 70% to 85%) typically has more impact than improving the ranking model.
