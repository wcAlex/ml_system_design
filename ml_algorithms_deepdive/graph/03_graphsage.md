# GraphSAGE: Scalable and Inductive Graph Learning

## The Problem GraphSAGE Solves

GCN has two fundamental limitations:

1. **Transductive**: requires the full graph at training time. Can't handle new nodes.
2. **Scalability**: full-batch training (every node, every edge) is infeasible at billion-node scale.

GraphSAGE (Graph **SA**mple and aggre**G**at**E**, Hamilton et al. 2017) was developed at Stanford to power Pinterest's graph at scale. It solves both problems.

---

## The Key Insight: Learn a Function, Not Just Embeddings

GCN learns an embedding for **each specific node**. If a new user joins tomorrow, there's no embedding for them.

GraphSAGE learns a **function** that takes a node's features and neighborhood features as input and produces an embedding. Apply that function to any node — including new ones — and you get an embedding.

```
GCN:    node_id → embedding (lookup table)
SAGE:   (node_features, neighbor_features) → embedding (learned function)
```

This is the difference between **transductive** and **inductive** learning.

---

## How GraphSAGE Works

### Step 1: Sample Neighbors (the "Sample" in GraphSAGE)

Instead of using ALL neighbors (which could be thousands), randomly sample a fixed number.

```
Node C has 1000 neighbors
GraphSAGE: randomly sample 25 neighbors each time
```

This gives a fixed-size input regardless of node degree, and makes mini-batch training possible.

### Step 2: Aggregate Neighbor Features (the "Aggregate" in GraphSAGE)

Take the sampled neighbors' features and aggregate them into a single vector.

GraphSAGE offers three aggregator options (more on these below):
- Mean aggregator
- LSTM aggregator
- Max-pooling aggregator

```
Neighbors of C (sampled): [h_A, h_B, h_D, h_E]
Aggregated:  agg_C = Aggregate([h_A, h_B, h_D, h_E])
```

### Step 3: Concatenate and Transform

Concatenate the node's own features with the aggregated neighbor vector, then apply a linear layer:

```
h_C^(l+1) = σ( W · concat(h_C^(l), agg_C) )
```

This is different from GCN's averaging — the node's own identity is preserved separately from its neighborhood context.

### Step 4: Normalize

Normalize the output to unit norm (optional but often helps):

```
h_C^(l+1) = h_C^(l+1) / ||h_C^(l+1)||
```

### Multi-hop: Sample Hierarchically

For 2-layer GraphSAGE, sample neighbors of neighbors:

```
To embed node C with 2 hops, depth-first:
  - Sample 25 neighbors of C: [A, B, D, ...]
  - For each of those, sample 10 neighbors: [A's neighbors: X, Y, Z, ...]
  - Aggregate from the bottom up
```

This mini-batch contains only the nodes actually needed — not the whole graph.

---

## The Three Aggregators

### 1. Mean Aggregator (simple, effective)

```
agg_C = mean([h_A, h_B, h_D, h_E])
```

- Essentially the same as GCN (but with sampling)
- Fast and usually works well
- Treats all sampled neighbors equally

### 2. LSTM Aggregator

```
agg_C = LSTM([h_A, h_B, h_D, h_E])  # random ordering
```

- More expressive — can capture sequential patterns in the neighborhood
- Problem: LSTM assumes order, but graph neighbors have no natural order
- Solution: randomly shuffle neighbors each time (acts as data augmentation)
- Slower, sometimes better on heterogeneous neighborhoods

### 3. Max-Pooling Aggregator (recommended in the paper)

```
agg_C = max( [σ(W_pool · h_v) for v in neighbors] )  # element-wise max
```

- First apply a neural network to each neighbor independently
- Then take the element-wise maximum
- Captures the "most important signal" across neighbors
- Often the best performer in practice

---

## Inductive Learning: Handling New Nodes

This is GraphSAGE's killer feature.

Scenario: Your social network has 100M users. A new user signs up.

**GCN**: No embedding. Must retrain the entire model with the new node included.

**GraphSAGE**:
1. New user fills out their profile → node features
2. New user connects to some existing users → edges
3. Sample neighbors, aggregate their features
4. Run the learned aggregation function
5. Get an embedding immediately. No retraining.

This is critical for production systems where graphs change every second.

---

## Mini-batch Training

GraphSAGE enables **mini-batch gradient descent** on graphs, which GCN can't do efficiently.

```
Mini-batch of target nodes: [C, D, F, G]  (batch_size = 4)

For each target node:
  - Sample its neighbors
  - Sample neighbors' neighbors (for 2-layer)
  - Only load these nodes into memory

Memory: O(batch_size × fanout^depth) instead of O(all_nodes)
```

For batch_size=256, fanout=25, depth=2:
- Memory: 256 × 25 × 25 = 160,000 node features
- vs. GCN requiring all N nodes

This is how Pinterest ran GraphSAGE on a graph with **3 billion nodes** (PinSage system).

---

## PinSage: GraphSAGE at Pinterest Scale

Pinterest implemented GraphSAGE as "PinSage" (2018) — the largest graph neural network deployed at the time.

**Graph**: 3 billion pins (images), 18 billion edges (user engagement)

**Key modifications for scale**:
1. **Random walk-based neighbor sampling**: instead of uniform sampling, use random walks to find the most important neighbors (nodes visited most often in random walks from the target)
2. **Hard negative mining**: training pairs where similar-looking pins that are semantically different force the model to learn fine-grained distinctions
3. **Curriculum training**: start with easy negatives, graduate to hard ones
4. **GPU-based inference**: batch inference for all 3 billion pins weekly

**Result**: +25% improvement in recommendation quality vs. previous systems.

---

## Trade-offs

### GraphSAGE vs. GCN

| Aspect | GCN | GraphSAGE |
|--------|-----|-----------|
| Inductive (new nodes) | No | Yes |
| Mini-batch training | Hard | Yes |
| Scale | Medium | Large (billions) |
| Neighbor weighting | Equal (normalized degree) | Equal (among sampled) |
| Neighbor sampling | No (all neighbors) | Yes (fixed fanout) |
| Aggregation options | Only mean | Mean, LSTM, max-pool |

### GraphSAGE Weaknesses

| Weakness | Detail |
|----------|--------|
| **Random sampling** | Dropping neighbors introduces noise; important neighbors might be missed |
| **Equal neighbor weights** | Like GCN, doesn't learn *which* neighbors matter more (GAT fixes this) |
| **Fixed fanout** | High-degree nodes (celebrities with millions of followers) are heavily subsampled |
| **Over-smoothing** | Still present, though less severe due to concatenation |
| **Heterogeneous graphs** | Same aggregation for all edge types — doesn't distinguish "friend" from "colleague" |

---

## When to Use GraphSAGE

| Scenario | Recommendation |
|----------|----------------|
| New nodes added frequently | ✅ Best choice (inductive) |
| Billion-scale graph | ✅ Mini-batch enables this |
| Pinterest-style item recommendation | ✅ PinSage is based on this |
| Need to weight neighbors by importance | ❌ Use GAT |
| Multiple edge types (LinkedIn) | ❌ Use HGT or R-GCN |
| Small static graph | ⚠️ GCN is simpler and works fine |

---

## Key Interview Points

1. **Why is GraphSAGE inductive but GCN transductive?**
   - GCN learns node-specific embeddings (lookup table)
   - GraphSAGE learns an aggregation *function* applicable to any node

2. **How does GraphSAGE scale to billions of nodes?**
   - Neighbor sampling → fixed computation per node
   - Mini-batch training → only load the nodes you need

3. **What is the fanout hyperparameter?**
   - Number of neighbors sampled per node per layer
   - Typical values: 25 for layer 1, 10 for layer 2
   - Higher = better approximation, higher memory/compute

4. **What's the tradeoff of sampling neighbors vs. using all?**
   - Sampling: O(fanout) computation, but noisy
   - All neighbors: exact but O(degree), infeasible for hubs

5. **How is GraphSAGE used in production at Pinterest?**
   - Offline: embed all 3B pins weekly
   - Online: query ANN index (FAISS/ScaNN) for nearest neighbors
   - Re-rank with user context

---

## Next

See `04_gat.md` — Graph Attention Networks, which fix the equal-weighting problem by learning which neighbors matter most.
