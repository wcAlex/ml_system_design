# Approximate Nearest Neighbors (ANN)

## 1. What Is ANN?

ANN is a family of algorithms and data structures that find the **approximately** closest points to a query, trading a small amount of accuracy for **orders-of-magnitude speed improvement** over exact KNN.

```
Exact KNN (brute-force):     ANN:
  Search ALL 10M items         Search a smart subset
  Time: ~100ms                 Time: ~1ms
  Recall: 100%                 Recall: 95-99%
  ❌ Too slow for serving      ✅ Production-ready
```

**Why approximate?** In recommendation systems with 10M+ items, exact KNN requires computing distance to every item. At 50K QPS, that's 500 billion distance computations per second — infeasible. ANN gives us 95-99% of the quality at 1/100th the cost.

**The core trade-off:**

```
                    Recall (quality)
                    ▲
              100%  │ ● Exact KNN
                    │
               98%  │         ● HNSW
               95%  │    ● IVF-Flat
               90%  │  ● IVF-PQ
                    │
                    └───────────────────▶ Queries/second (speed)
                       1K    10K   100K
```

---

## 2. Major ANN Algorithm Families

### Family 1: Tree-Based (KD-Tree, Annoy)

**Idea**: Recursively partition the space with hyperplanes. At query time, only search the partition the query falls into (plus a few nearby ones).

```
2D example with random projection trees:

         ┌─────────────────────┐
         │   ●  ●              │
         │  ●     ─────────    │  Split 1 (horizontal)
         │         ●   ●      │
         │  ●  ●  │  ●        │  Split 2 (vertical)
         │        │     ●     │
         └─────────────────────┘

Query lands in bottom-right cell → search only that cell
+ check neighboring cells for border cases
```

**Algorithm (Annoy — used by Spotify):**
1. Build multiple random projection trees (forest)
2. Each tree recursively splits data with random hyperplanes
3. At query time, traverse each tree to find candidate leaf nodes
4. Union candidates from all trees, compute exact distances on that subset
5. Return top-K

```python
# Annoy example (Spotify's library)
from annoy import AnnoyIndex

dim = 128
num_trees = 50  # More trees = better recall, more memory

# Build index
index = AnnoyIndex(dim, metric="angular")  # angular ≈ cosine
for i, embedding in enumerate(all_video_embeddings):
    index.add_item(i, embedding)
index.build(num_trees)
index.save("video_index.ann")

# Query
index = AnnoyIndex(dim, metric="angular")
index.load("video_index.ann")
neighbor_ids, distances = index.get_nns_by_vector(
    query_embedding, n=100, include_distances=True
)
```

**Characteristics:**
- Build time: O(N × T × log N), where T = number of trees
- Query time: O(T × log N + |candidates| × D)
- Memory: O(N × D × T)
- Recall improves with more trees (but more memory)

### Family 2: Hash-Based (Locality-Sensitive Hashing / LSH)

**Idea**: Design hash functions where similar items are likely to hash to the same bucket. At query time, only compare items in the same bucket.

```
Hash function for cosine similarity:
  h(x) = sign(r · x)  where r is a random vector

Similar vectors → likely same sign → same bucket

Example with 3 hash functions:
  video_A → hash = (1, 0, 1) → bucket "101"
  video_B → hash = (1, 0, 1) → bucket "101"  ← same bucket, likely similar
  video_C → hash = (0, 1, 0) → bucket "010"  ← different bucket
```

**Algorithm:**
1. Generate L sets of K random hash functions
2. For each item, compute L hash signatures, store in L hash tables
3. At query time: compute L hash signatures, collect all items from matching buckets
4. Compute exact distance on collected candidates, return top-K

```python
# Conceptual LSH implementation
import numpy as np


class LSH:
    """Locality-Sensitive Hashing for cosine similarity."""

    def __init__(self, dim: int, num_tables: int = 10, num_hashes: int = 8):
        self.num_tables = num_tables
        self.num_hashes = num_hashes
        # Random hyperplanes for hashing
        self.hyperplanes = [
            np.random.randn(num_hashes, dim) for _ in range(num_tables)
        ]
        # Hash tables: table_idx → {hash_key → [item_ids]}
        self.tables = [{} for _ in range(num_tables)]
        self.data = {}  # id → vector

    def _hash(self, vector: np.ndarray, table_idx: int) -> str:
        projections = self.hyperplanes[table_idx] @ vector
        bits = (projections > 0).astype(int)
        return "".join(map(str, bits))

    def add(self, item_id: int, vector: np.ndarray):
        self.data[item_id] = vector
        for t in range(self.num_tables):
            key = self._hash(vector, t)
            self.tables[t].setdefault(key, []).append(item_id)

    def query(self, vector: np.ndarray, top_k: int = 100) -> list:
        # Collect candidates from all tables
        candidates = set()
        for t in range(self.num_tables):
            key = self._hash(vector, t)
            candidates.update(self.tables[t].get(key, []))

        # Exact distance on candidates only
        scored = []
        for cid in candidates:
            dist = np.dot(vector, self.data[cid]) / (
                np.linalg.norm(vector) * np.linalg.norm(self.data[cid]) + 1e-10
            )
            scored.append((cid, dist))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
```

**Characteristics:**
- Build time: O(N × L × K)
- Query time: O(L × K + |candidates| × D)
- Memory: O(N × L)
- Recall improves with more tables (L) and fewer hashes per table (K)

### Family 3: Graph-Based (HNSW)

**Idea**: Build a navigable graph where each node is connected to its approximate nearest neighbors. At query time, greedily traverse the graph toward the query point.

**HNSW (Hierarchical Navigable Small World)** — the current state of the art for recall/speed trade-off.

```
HNSW multi-layer structure:

Layer 2 (few nodes, long-range links):
  A ─────────────────── D
  │                     │

Layer 1 (more nodes, medium-range links):
  A ──── B ──── C ──── D ──── E
  │      │      │      │      │

Layer 0 (all nodes, short-range links):
  A ─ B ─ C ─ D ─ E ─ F ─ G ─ H ─ I ─ J
  │   │   │   │   │   │   │   │   │   │
  (fine-grained connections)

Query traversal:
  1. Enter at Layer 2, greedily move to nearest node → land near D
  2. Drop to Layer 1, refine → find C is closer
  3. Drop to Layer 0, refine → find exact neighborhood

Similar to express/local stops on a subway:
  Layer 2 = express stops (skip large areas fast)
  Layer 0 = local stops (search precisely)
```

**Algorithm:**
1. **Build**: Insert each point, connecting it to its nearest neighbors at each layer. Higher layers are sparser (nodes promoted with probability p=1/M).
2. **Query**: Start at top layer, greedily walk toward query. At each layer, find the closest entry point, then descend. At layer 0, expand search to find K nearest neighbors.

```python
# HNSW via hnswlib
import hnswlib

dim = 128
num_items = 10_000_000

# Build index
index = hnswlib.Index(space="ip", dim=dim)  # "ip" = inner product (cosine after L2-norm)
index.init_index(
    max_elements=num_items,
    ef_construction=200,  # Build-time search width (higher = better graph, slower build)
    M=32,                 # Max connections per node (higher = better recall, more memory)
)

# Add items in batches
for batch_start in range(0, num_items, 100_000):
    batch_embs = load_embeddings(batch_start, batch_start + 100_000)
    batch_ids = list(range(batch_start, batch_start + 100_000))
    index.add_items(batch_embs, batch_ids)

# Set query-time search parameter
index.set_ef(100)  # Query-time search width (higher = better recall, slower query)

# Query
labels, distances = index.knn_query(query_embedding, k=100)
```

**Characteristics:**
- Build time: O(N × log N × M)
- Query time: O(log N × ef × D), where ef = search width
- Memory: O(N × (D + M × layers))
- **Best recall/speed trade-off** among all ANN methods

### Family 4: Quantization-Based (IVF, PQ, ScaNN)

**Idea**: Compress vectors to reduce memory and distance computation cost. Two techniques are often combined:

#### IVF (Inverted File Index)

Partition the space into clusters (using K-means). At query time, only search the nearest clusters.

```
Cluster the space into 1000 cells via K-means:

  ┌─────┬─────┬─────┐
  │  C₁  │  C₂  │  C₃  │
  │ ●●●  │ ●●   │ ●●●● │
  │  ●   │ ●●●  │  ●●  │
  ├─────┼─────┼─────┤
  │  C₄  │  C₅  │  C₆  │
  │ ●●   │ ●●●  │ ●●   │
  │  ●●  │  ★Q  │  ●   │  ★ = query
  ├─────┼─────┼─────┤
  │  C₇  │  C₈  │  C₉  │
  │ ●●●● │ ●●   │ ●●●  │
  │  ●   │  ●   │  ●   │
  └─────┴─────┴─────┘

Query Q lands near C₅ → search C₅ and its neighbors (nprobe=3: C₂, C₅, C₈)
Skip the other 997 cells entirely
```

#### PQ (Product Quantization)

Compress each 128-d vector into a short code (e.g., 32 bytes) by splitting the vector into sub-vectors and quantizing each.

```
Original vector (128 floats = 512 bytes):
  [0.12, -0.34, 0.56, ..., 0.78]

Split into 32 sub-vectors of 4 dimensions each:
  [0.12, -0.34, 0.56, 0.01] | [-0.23, 0.45, 0.11, -0.67] | ...

Each sub-vector → nearest centroid ID (1 byte):
  [42] | [17] | [88] | ...

Compressed code (32 bytes instead of 512):
  [42, 17, 88, ...]

Distance computed via lookup table → very fast
```

#### FAISS (Facebook AI Similarity Search)

The most widely used ANN library. Supports all index types.

```python
import faiss
import numpy as np

dim = 128
num_items = 10_000_000
embeddings = load_all_embeddings()  # (10M, 128) float32

# ─── Option 1: Flat (exact, for reference) ───
index_flat = faiss.IndexFlatIP(dim)
index_flat.add(embeddings)
# Memory: 10M × 128 × 4 = 4.88 GB
# Query: ~100ms (too slow for production)

# ─── Option 2: IVF-Flat (partitioned exact) ───
nlist = 4096  # number of clusters
quantizer = faiss.IndexFlatIP(dim)
index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
index_ivf.train(embeddings[:500_000])  # Train K-means on a sample
index_ivf.add(embeddings)
index_ivf.nprobe = 32  # Search 32 out of 4096 clusters
# Memory: ~5 GB
# Query: ~5ms, Recall@100 ~95-98%

# ─── Option 3: IVF-PQ (partitioned + compressed) ───
m = 32  # number of sub-quantizers
index_ivfpq = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)
index_ivfpq.train(embeddings[:500_000])
index_ivfpq.add(embeddings)
index_ivfpq.nprobe = 64
# Memory: ~1.2 GB (huge savings!)
# Query: ~2ms, Recall@100 ~90-95%

# ─── Option 4: HNSW + Flat ───
index_hnsw = faiss.IndexHNSWFlat(dim, 32)  # M=32
index_hnsw.hnsw.efConstruction = 200
index_hnsw.add(embeddings)
index_hnsw.hnsw.efSearch = 100
# Memory: ~8 GB (graph overhead)
# Query: ~1ms, Recall@100 ~97-99%

# ─── Query (same for all) ───
query = np.random.randn(1, dim).astype("float32")
faiss.normalize_L2(query)  # For cosine similarity via inner product
scores, indices = index.search(query, k=100)
```

#### Google ScaNN

Google's ANN library, uses **anisotropic quantization** — accounts for the direction of the quantization error, not just its magnitude.

```python
import scann

searcher = scann.scann_ops_pybind.builder(
    embeddings, num_neighbors=100, distance_measure="dot_product"
).tree(
    num_leaves=2000,          # IVF partitions
    num_leaves_to_search=100  # nprobe
).score_ah(
    dimensions_per_block=2,   # Anisotropic hashing
    anisotropic_quantization_threshold=0.2
).reorder(
    reorder_num_neighbors=200  # Re-rank top candidates with exact distance
).build()

neighbors, distances = searcher.search(query_embedding, final_num_neighbors=100)
```

---

## 3. Algorithm Comparison

| Algorithm | Query Speed | Recall@100 | Memory | Build Time | Best For |
|-----------|-----------|------------|--------|------------|----------|
| **Brute-force** | Slow (~100ms) | 100% | N×D×4B | None | < 100K items, testing |
| **Annoy** | Fast (~5ms) | 90-95% | N×D×4B × T | Medium | Read-heavy, Spotify-scale |
| **LSH** | Fast (~5ms) | 85-95% | N × L tables | Fast | Streaming data, simple setup |
| **HNSW** | Fastest (~1ms) | 97-99% | N×D×4B + graph | Slow | Best quality, memory available |
| **IVF-Flat** | Fast (~5ms) | 95-98% | N×D×4B | Medium | Good all-rounder |
| **IVF-PQ** | Fastest (~2ms) | 90-95% | N×m bytes | Medium | Memory-constrained, huge datasets |
| **ScaNN** | Fastest (~1ms) | 95-98% | Compressed | Medium | Google stack, best quantization |

### Decision Tree

```
How many items?
│
├── < 100K items → Brute-force (exact KNN via FAISS Flat)
│
├── 100K - 1M items
│   ├── Need best recall? → HNSW
│   └── Memory limited? → IVF-Flat
│
├── 1M - 100M items
│   ├── Need best recall? → HNSW (if memory allows: ~8B per item)
│   ├── Balanced? → IVF-Flat (nprobe tuning)
│   └── Memory limited? → IVF-PQ (saves 4-10x memory)
│
└── > 100M items
    ├── Distributed? → Shard IVF across machines
    ├── Google stack? → ScaNN
    └── Managed service? → Pinecone, Vertex AI Vector Search
```

---

## 4. Worked Example: Video Retrieval with FAISS

### Setup

```
Corpus: 5M video embeddings, 128-d, L2-normalized
Goal: For each user, retrieve top-100 similar videos in < 10ms
```

### Step-by-Step

```python
import faiss
import numpy as np

# 1. Load video embeddings
video_embeddings = np.load("video_embs.npy")  # (5_000_000, 128), float32
video_ids = np.load("video_ids.npy")           # (5_000_000,), string

# 2. Choose index: IVF-Flat (good balance for 5M items)
dim = 128
nlist = 2048  # sqrt(5M) ≈ 2236, round to power of 2

quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

# 3. Train the index (K-means clustering)
# Use a 10% sample for training
train_sample = video_embeddings[np.random.choice(len(video_embeddings), 500_000, replace=False)]
index.train(train_sample)

# 4. Add all vectors
index.add(video_embeddings)

# 5. Tune nprobe (trade recall for speed)
# nprobe=1:   ~0.5ms, Recall≈70%
# nprobe=16:  ~3ms,   Recall≈92%
# nprobe=32:  ~5ms,   Recall≈96%  ← good trade-off
# nprobe=128: ~15ms,  Recall≈99%
index.nprobe = 32

# 6. Query: user embedding from Two-Tower model
user_embedding = two_tower_model.user_tower(user_features)  # (1, 128)
user_embedding = user_embedding / np.linalg.norm(user_embedding)  # L2 normalize

scores, indices = index.search(user_embedding, k=100)

# 7. Map back to video IDs
results = [
    {"video_id": video_ids[idx], "score": float(score)}
    for idx, score in zip(indices[0], scores[0])
    if idx >= 0  # FAISS returns -1 for missing results
]

print(f"Retrieved {len(results)} candidates")
print(f"Top-5: {results[:5]}")
```

**Output:**
```
Retrieved 100 candidates
Top-5:
  {"video_id": "v_48291", "score": 0.923}
  {"video_id": "v_10382", "score": 0.891}
  {"video_id": "v_77234", "score": 0.887}
  {"video_id": "v_30291", "score": 0.874}
  {"video_id": "v_55192", "score": 0.861}
```

---

## 5. Applied Scenarios in Recommendation Systems

### Scenario 1: Two-Tower Retrieval (Primary Use Case)

```
Two-Tower model produces:
  user_embedding (128-d) — computed at request time
  item_embeddings (128-d each) — precomputed for all 10M videos

ANN's role:
  Given user_embedding, find top-100 items by inner product
  Using FAISS IVF or HNSW → < 5ms

This is the CORE retrieval mechanism at YouTube, Instagram, Pinterest
```

### Scenario 2: Content-Based Retrieval

```
Sentence-transformer encodes video titles → 128-d embeddings
User's profile = weighted average of recent watch history embeddings

ANN's role:
  Given user_content_profile, find top-100 content-similar videos
  Handles cold-start (new videos with no interactions)
```

### Scenario 3: "Similar Videos" Feature

```
User is watching video V → show "More like this" sidebar

ANN's role:
  Given V's embedding, find top-20 nearest videos
  Single vector query → instant results
```

### Scenario 4: Deduplication / Near-Duplicate Detection

```
New video uploaded → check if it's a duplicate/re-upload

ANN's role:
  Given new video's embedding, find nearest neighbor
  If similarity > 0.95 → flag as potential duplicate
  Runs as part of the upload pipeline
```

### Scenario 5: Clustering for Exploration

```
Cluster all video embeddings into 1000 topic clusters
Pick exploration candidates from clusters the user hasn't seen

ANN's role (inverted):
  The IVF index already has clusters from K-means
  Use those cluster assignments for topic-based exploration
```

---

## 6. Production Considerations

### Index Updates for New Content

```
Problem: New videos uploaded hourly, but main index is rebuilt every 4-6 hours
Solution: Dual-index architecture

  ┌──────────────────┐     ┌──────────────────┐
  │  Main Index       │     │  Fresh Index      │
  │  10M videos       │     │  50K videos       │
  │  Rebuilt every 6h  │     │  Rebuilt every 1h  │
  │  IVF-PQ           │     │  Flat (exact)      │
  └────────┬─────────┘     └────────┬─────────┘
           │                        │
           └──── query both, merge results ────┘
                         │
                   top-K combined
```

### Sharding for Very Large Indices

```
100M items won't fit on one machine (HNSW: ~100M × 200B ≈ 20 GB + embeddings)

Sharding strategies:
  Option A: Shard by hash(video_id) → each shard holds N/S items
            Query ALL shards in parallel, merge top-K
            Pro: Even distribution
            Con: Must query all shards (fan-out)

  Option B: Shard by cluster → each shard holds items from specific clusters
            Route query to relevant shards only
            Pro: Lower fan-out
            Con: Uneven shard sizes

  Option C: Replicate small enough index
            If index fits in memory per machine, just replicate for throughput
            Pro: Simplest
            Con: Memory cost multiplied
```

### Recall Measurement in Production

```python
def measure_ann_recall(index, exact_index, queries, k=100):
    """
    Compare ANN results against exact brute-force results.
    Run periodically on a sample of queries.
    """
    recalls = []
    for query in queries:
        ann_scores, ann_ids = index.search(query.reshape(1, -1), k)
        exact_scores, exact_ids = exact_index.search(query.reshape(1, -1), k)

        ann_set = set(ann_ids[0])
        exact_set = set(exact_ids[0])

        recall = len(ann_set & exact_set) / k
        recalls.append(recall)

    avg_recall = np.mean(recalls)
    print(f"ANN Recall@{k}: {avg_recall:.3f}")
    return avg_recall

# Target: Recall@100 > 0.95
```

---

## 7. Key Trade-Offs Summary

| Decision | Option A | Option B | Guidance |
|----------|----------|----------|----------|
| **Index type** | HNSW (best recall) | IVF-PQ (lowest memory) | HNSW if memory allows; IVF-PQ for 100M+ items |
| **nprobe / ef** | Low (fast, low recall) | High (slow, high recall) | Tune to hit recall target at latency budget |
| **Rebuild frequency** | Frequent (fresh) | Infrequent (stable) | Main index every 4-6h, fresh index every 1h |
| **Exact reranking** | Yes (fetch 200, exact-score top 100) | No (trust ANN scores) | Yes, if you can afford the latency |
| **Library** | FAISS (self-hosted) | Managed (Pinecone, Vertex) | FAISS for control/cost; managed for simplicity |

---

## 8. Connection to Video Recommendation Pipeline

ANN is the **engine** that makes the retrieval stage feasible at scale:

```
Without ANN:
  10M videos × 128-d × 50K QPS = too slow

With ANN (FAISS IVF, nprobe=32):
  Search 32/2048 clusters × ~2400 items/cluster × 128-d = fast
  ~5ms per query → 50K QPS on a few machines

Pipeline integration:
  Two-Tower Model → user_embedding → FAISS search → 100 candidates ─┐
  Content Model → user_profile → FAISS search → 100 candidates ─────┤
  Popularity → Redis sorted set → 50 candidates ────────────────────┤
  Subscriptions → filter recent uploads → 50 candidates ────────────┤
                                                                     ▼
                                                          Merge → ~300 unique
                                                                     │
                                                              Ranking Model
                                                                     │
                                                               Re-Ranking
                                                                     │
                                                            30 recommendations
```

ANN directly enables the **sub-10ms retrieval latency** required for the serving system's 200ms total budget (see `05_serving_system.md`).
