# 06 — Model Development

The PYMK pipeline has two stages. Each stage uses a different model with different objectives.

```
Stage 1: Candidate Generation   (billions → ~1,000 candidates)
Stage 2: Ranking                (1,000 candidates → top 20)
```

---

## Stage 1: Candidate Generation

### Option A: Graph Traversal (Baseline)

**Algorithm:**
1. Fetch all 1st-degree connections of the query user (avg 1,000)
2. For each 1st-degree connection, fetch their connections (1,000 × 1,000 = up to 1M)
3. Remove: already connected, already dismissed, the query user themselves
4. Sort by `common_neighbor_count` descending
5. Keep top 2,000

**Pros:** Simple, interpretable, always produces a meaningful "mutual connections" reason code
**Cons:** 2nd-degree only; misses people with similar profiles but no graph overlap; O(D²) per user where D = avg degree

### Option B: ANN Search on GNN Embeddings (Recommended)

**Algorithm:**
1. Offline: train GNN, produce 128-d embedding for every user
2. Offline: build an Approximate Nearest Neighbor (ANN) index over all 1B embeddings
3. Online: retrieve query user's embedding, search ANN index for top-2,000 nearest neighbors

**ANN Index options:**

| Method | Algorithm | Recall@100 | Latency | Memory |
|---|---|---|---|---|
| FAISS (Facebook) | HNSW / IVF-PQ | 95%+ | 1–10ms | Moderate |
| ScaNN (Google) | Anisotropic quantization | 97%+ | ~1ms | Low |
| HNSW (nmslib) | Hierarchical NSW graph | 98%+ | ~5ms | High |

For 1B users × 128-d float32 = 512GB raw; with quantization (int8) = 128GB — fits on a cluster of machines.

**Pros:** Captures semantic similarity beyond graph structure; fast at serving time; handles cold-start better (profile-based embedding)
**Cons:** Requires GNN training and index build; ANN has approximate recall (misses some candidates)

### Option C: Hybrid (Production Best Practice)

Merge results from both Option A and Option B:
```
candidates = union(graph_traversal_top_1000, ann_search_top_1000)
             → deduplicate
             → filter (already connected, dismissed, blocked)
             → keep top 2,000 by rough heuristic score
```

Graph traversal captures strong structural candidates; ANN captures semantically similar but structurally distant users. The union covers both cases.

---

## Stage 2: Ranking

### Option A: Gradient Boosted Trees (GBT) — Good Baseline

**Why GBT works well here:**
- Handles mixed feature types natively (floats, integers, categoricals, binary flags)
- Robust to missing values
- Extremely interpretable (feature importance, SHAP values)
- Trains fast

**Architecture:**
```python
features = [
    # graph features
    "common_neighbor_count", "adamic_adar_score", "jaccard_sim",
    "graph_distance_bucket", "query_degree_log", "candidate_degree_log",
    # profile features
    "same_company_current", "same_school", "same_industry",
    "skill_jaccard_sim", "title_cosine_sim", "geo_distance_km",
    # behavioral features
    "query_viewed_candidate_profile", "profile_view_recency_days",
    "co_engaged_with_content", "mutual_profile_views",
    # embedding features
    "gnn_embedding_cosine_sim",
]

model = XGBoost(
    objective="binary:logistic",
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
)
```

**Training objective:** Binary cross-entropy on (query_user, candidate, label) triplets
**Label:** 1 = connection accepted, 0 = impression without action after 14 days

**Pros:** Fast training/inference, interpretable, good baseline
**Cons:** Cannot learn complex feature interactions automatically; embedding features are reduced to a single scalar (cosine sim)

### Option B: Two-Tower Neural Model with GNN (Recommended)

**Architecture:**

```
Query Tower                         Candidate Tower
───────────────────────────────     ───────────────────────────────
Input:                              Input:
  - GNN embedding (128-d)             - GNN embedding (128-d)
  - profile features (32-d)           - profile features (32-d)
  - activity features (16-d)          - activity features (16-d)

FC(176, 256) + BatchNorm + ReLU     FC(176, 256) + BatchNorm + ReLU
FC(256, 128) + BatchNorm + ReLU     FC(256, 128) + BatchNorm + ReLU
FC(128, 64)                         FC(128, 64)

L2-normalize → q_embed (64-d)       L2-normalize → c_embed (64-d)

                        dot(q_embed, c_embed)
                                │
                        sigmoid → P(connect)
```

**GNN Component (GraphSAGE-style):**

```python
class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(2 * in_dim, out_dim)  # concat self + agg

    def forward(self, self_feat, neighbor_feats):
        # neighbor_feats: (batch, num_samples, in_dim)
        agg = neighbor_feats.mean(dim=1)           # mean aggregator
        combined = torch.cat([self_feat, agg], dim=-1)
        return F.relu(self.W(combined))

class GraphSAGE(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, embed_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphSAGELayer(node_feat_dim if i == 0 else hidden_dim,
                          hidden_dim if i < num_layers-1 else embed_dim)
            for i in range(num_layers)
        ])

    def forward(self, node_features, neighbor_features_list):
        h = node_features
        for layer, neighbor_feats in zip(self.layers, neighbor_features_list):
            h = layer(h, neighbor_feats)
        return h  # node embedding
```

**Loss Function:**

Two options for training the two-tower model:

**Option B1: Binary Cross-Entropy**
```python
loss = BCELoss(sigmoid(dot(q, c)), label)
```
Simple, but requires balanced sampling.

**Option B2: In-Batch Negative Sampling (recommended, more efficient)**
```python
# Batch: N positive (query, candidate) pairs
# Use all other N-1 candidates in the batch as negatives for each query
scores = torch.matmul(q_embeds, c_embeds.T)  # (N, N) score matrix
# Diagonal = positive pairs, off-diagonal = negatives
labels = torch.eye(N)
loss = CrossEntropyLoss(scores, labels.argmax(dim=1))
```
This is **sampled softmax loss**, used by Google (DPR), YouTube, LinkedIn. It leverages the batch structure to get N-1 negatives per example at no extra cost.

**Option B3: Triplet Loss with Hard Negative Mining**
```python
# For each (anchor=query, positive=accepted candidate):
# Find hard negative = highest-scoring non-connected candidate
loss = max(0, margin - score(q, pos) + score(q, hard_neg))
```
Hard negatives accelerate training convergence.

### Option C: End-to-End GNN Ranker (Cross-Attention, Advanced)

Instead of two separate towers, use a **cross-attention** mechanism that directly models the interaction between query and candidate subgraphs.

```
query_subgraph + candidate_subgraph
         │
    [Cross-Attention GNN]
    (each node attends to both subgraphs)
         │
    [Pooling]
         │
    connection_score
```

**Pros:** Captures interaction-level features (e.g., "the path between A and B goes through their shared boss")
**Cons:** Cannot precompute candidate embeddings — must run at serving time per pair. Not scalable for Stage 2 on 1,000 candidates unless batched efficiently.

Used for re-ranking the top 50 after the two-tower model.

---

## Training Setup

### Data Pipeline
```
1. Sample training triplets: (query_user, positive_candidate, negative_candidate)
2. For each node: sample k-hop neighborhood (e.g., 10 neighbors × 2 hops)
3. Fetch node features for all sampled nodes
4. Batch size: 512 triplets
5. Shuffle across days to prevent temporal autocorrelation
```

### Negative Sampling Strategy

| Strategy | When to use | Notes |
|---|---|---|
| Random negatives | Early training | Easy negatives, fast convergence start |
| Popularity-weighted | Mid training | Sample negatives proportional to degree |
| Hard negatives (in-batch) | Late training | Use model's own predictions to find hard cases |

**Curriculum learning** (PinSage approach): Start with easy negatives, progressively introduce harder ones as training progresses. Prevents model from getting stuck in local minima early.

### Hyperparameters
```
GNN:
  - layers: 2 (captures 2-hop neighborhoods)
  - hidden dim: 256
  - embedding dim: 128
  - neighbor samples per hop: 10
  - dropout: 0.3

Two-Tower:
  - tower hidden dims: [256, 128, 64]
  - batch size: 2048
  - learning rate: 1e-3 with cosine decay
  - weight decay: 1e-5
  - training epochs: 10–20 (with early stopping on validation AUC)
```

---

## GNN Training at Scale

Training a GNN on 1B nodes requires distributed training. Two approaches:

### Approach 1: Mini-batch Subgraph Sampling (Recommended)
```python
# For each training batch:
# 1. Sample M seed nodes (query users)
# 2. For each seed, expand k-hop neighborhood via random walks
# 3. Build mini-graph containing only sampled nodes
# 4. Run GNN forward pass on mini-graph
# 5. Compute loss on seed node embeddings only
```
This is what GraphSAGE and PinSage use. Each mini-batch is a small subgraph, enabling training with standard GPU memory.

### Approach 2: Graph Partitioning (Alternative)
Split the full graph into partitions (e.g., by community detection), train on each partition separately. Requires careful handling of cross-partition edges.

---

## Interview Checkpoint

**Q: Why use in-batch negatives instead of random sampled negatives?**

In-batch negatives are efficient and produce harder negatives naturally — within a batch of positive pairs, the "negative" for query user A is a user who is a genuine positive for another query user B. These are much harder negatives than random strangers, which forces the model to learn finer-grained discrimination. Google's DPR paper showed in-batch negatives significantly outperform random negatives.

**Q: How many GNN layers should you use?**

Each GNN layer expands the receptive field by 1 hop. With D=1000 average degree:
- 1 layer: 1,000 neighbors
- 2 layers: 1,000,000 neighbors (exponential blowup)
- 3 layers: 1 billion neighbors — basically the whole graph

In practice, 2 layers is almost always the right choice for social graphs. Beyond 2 layers, the "over-smoothing" problem kicks in: all node embeddings converge toward the same mean, losing discriminative power.

**Q: What is over-smoothing in GNNs?**

Over-smoothing occurs when a GNN has too many layers. Each layer aggregates neighbor information, and with enough layers, every node's embedding becomes dominated by the global graph average. Nodes lose their individual characteristics. Solutions: dropout on graph edges, residual connections (skip connections), or limiting to 2–3 layers.
