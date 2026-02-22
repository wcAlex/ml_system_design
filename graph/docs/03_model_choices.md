# 03 — Model Choices

This document starts from first principles — simple graph heuristics — and builds up to state-of-the-art GNN-based approaches. Understanding the progression helps you reason about trade-offs in an interview.

---

## The Spectrum: Simple → Complex

```
Level 1: Graph Heuristics         (no ML, interpretable, fast)
Level 2: Graph Embeddings         (unsupervised ML, scalable)
Level 3: Graph Neural Networks    (supervised deep learning)
Level 4: Two-Tower + GNN          (production-grade, LinkedIn-style)
Level 5: LLM-Enhanced             (emerging, profile understanding)
```

---

## Level 1: Graph Heuristics

**Core idea:** Use structural properties of the graph to score how "likely" two nodes should be connected. No training required.

### Common Neighbors
```
score(A, B) = |N(A) ∩ N(B)|
```
If A and B share many mutual friends, they probably know each other.

### Jaccard Coefficient
```
score(A, B) = |N(A) ∩ N(B)| / |N(A) ∪ N(B)|
```
Normalizes by total neighborhood size — avoids bias toward high-degree nodes.

### Adamic-Adar Index
```
score(A, B) = Σ  1 / log(|N(z)|)   for z in N(A) ∩ N(B)
```
Weights mutual friends inversely by their degree. Sharing a rare mutual friend (a niche expert) is more informative than sharing a celebrity who connects everyone.

### Katz Index
```
score(A, B) = Σ β^l × (number of paths of length l from A to B)
```
Considers all paths, not just 2nd-degree. β < 1 dampens longer paths.

### When to use heuristics
- Baseline / sanity check
- Interpretable "reason codes" for the UI ("12 mutual connections")
- Fast filtering in candidate generation

### Limitations
- No personalization beyond graph structure
- Ignores profile similarity, interaction history, temporal signals
- High-degree nodes ("hubs") dominate — a celebrity shares many "common connections" with everyone

---

## Level 2: Graph Embeddings (Unsupervised)

**Core idea:** Learn a low-dimensional vector for every user such that users who are likely to connect are close in embedding space. Use random walks on the graph to generate training sequences.

### DeepWalk (2014)
1. Run random walks starting from each node → sequences of node IDs
2. Treat sequences like sentences, node IDs like words
3. Train Word2Vec (Skip-gram) on these sequences
4. Each node gets a dense embedding vector

```
Random Walk: A → C → F → B → D → ...
Word2Vec sees: "predict node F given context [C, B]"
Result: nodes in similar neighborhoods get similar embeddings
```

### Node2Vec (2016)
Extends DeepWalk with two parameters:
- **p** (return parameter): probability of revisiting the previous node (local exploration)
- **q** (in-out parameter): probability of moving to a farther node (global exploration)

By tuning p and q, you can learn embeddings that capture either **homophily** (nearby nodes are similar) or **structural equivalence** (nodes with similar roles, e.g., bridges, are similar).

### LINE (2015)
Explicitly optimizes two objectives:
- **1st-order proximity:** directly connected nodes should be similar
- **2nd-order proximity:** nodes with similar neighbors should be similar

Good for very large graphs since it doesn't require random walks.

### Comparison

| Method | Scalability | Captures structure | Supervised | Cold-start |
|---|---|---|---|---|
| DeepWalk | Good | Local neighborhoods | No | Poor |
| Node2Vec | Good | Flexible (local/global) | No | Poor |
| LINE | Excellent | 1st + 2nd order | No | Poor |
| Common Neighbors | Excellent | 2nd degree only | No | Poor |

**Cold-start problem:** If a user has no connections, random-walk methods produce no training signal. This is a key limitation.

---

## Level 3: Graph Neural Networks (GNNs)

**Core idea:** Instead of pre-computing a fixed embedding, a GNN **learns how to aggregate information from neighbors** to produce an embedding. This means:
1. The embedding captures richer structural patterns
2. Node features (profile attributes) are incorporated
3. The model can generalize to unseen nodes (inductive learning)

### How a GNN Works — The Message Passing Framework

At each layer, every node:
1. **Aggregates** messages from its neighbors
2. **Combines** aggregated messages with its own current representation
3. **Updates** its representation

```
Layer 0: Each node = its raw features (e.g., profile embedding)
Layer 1: Each node = summary of its 1-hop neighborhood
Layer 2: Each node = summary of its 2-hop neighborhood
...
Layer k: Each node = summary of its k-hop neighborhood
```

After k layers, node A's embedding captures A's entire k-hop subgraph.

```python
# Pseudocode for one GNN layer
for node v in graph:
    neighbor_messages = [h[u] for u in neighbors(v)]
    aggregated = AGGREGATE(neighbor_messages)   # mean, sum, max, LSTM, attention
    h_new[v] = UPDATE(h[v], aggregated)         # concat + linear layer
```

### GraphSAGE (Hamilton et al., 2017) — Recommended for PYMK

**Key innovation:** Sample a fixed number of neighbors instead of using all of them. This makes training feasible on billion-scale graphs.

```
For each node v:
  1. Sample K neighbors (e.g., K=10 at each hop)
  2. Aggregate sampled neighbor features
  3. Concatenate with v's own features
  4. Apply learned linear transform + ReLU
```

**Inductive:** Can produce embeddings for brand-new users (just aggregate their neighbors' embeddings). Solves cold-start partially.

**Industry use:** Stanford/Pinterest paper (PinSage) directly adapts GraphSAGE to web-scale recommendation.

### Graph Attention Network (GAT)

Instead of averaging all neighbors equally, GAT learns **attention weights** — some neighbors are more important than others.

```
attention(v, u) = softmax( LeakyReLU( a^T [W*h_v || W*h_u] ) )
h_new[v] = σ( Σ attention(v,u) × W × h_u )
```

Better for heterogeneous graphs where connection types vary in importance.

### PinSage (Pinterest, 2018) — Production GNN at Scale

Key optimizations that made GNN training viable at Pinterest's scale (3B nodes):
1. **Random walk-based neighbor sampling** — importance-sampled neighborhoods via random walks
2. **Curriculum training** — start with easy negatives, progressively harder
3. **Mini-batch training** — sample computation subgraphs
4. **Producer-consumer architecture** — CPU builds mini-batches while GPU trains

These same ideas apply directly to PYMK.

### Comparison

| GNN Variant | Key strength | Scalability | Best for |
|---|---|---|---|
| GCN | Simple, established | Medium | Small/medium graphs |
| GraphSAGE | Inductive, efficient | High | Large graphs, cold start |
| GAT | Attention weights | Medium-High | Heterogeneous connections |
| PinSage | Web-scale, production-proven | Very High | 1B+ node graphs |

---

## Level 4: Two-Tower + GNN (Recommended Production Architecture)

**Core idea:** Use GNN embeddings as inputs to a two-tower neural model. The two towers produce embeddings for the query user and the candidate user respectively, and the dot product (or cosine similarity) is the ranking score.

```
Query User                          Candidate User
     │                                    │
     ▼                                    ▼
[GNN Embedding]               [GNN Embedding]
[Profile Features]            [Profile Features]
[Activity Features]           [Activity Features]
     │                                    │
     ▼                                    ▼
[Query Tower]                 [Candidate Tower]
(MLP layers)                  (MLP layers)
     │                                    │
     ▼                                    ▼
  q_embed (128-d)              c_embed (128-d)
           │                      │
           └──── dot product ─────┘
                      │
                    score
```

**Advantages:**
- Query and candidate embeddings can be precomputed separately
- Fast ANN search at inference time
- Incorporates both graph structure (GNN) and content features (profile)

This is the architecture used by LinkedIn, Meta, Twitter for connection recommendation.

---

## Level 5: LLM-Enhanced

**Emerging approach:** Use an LLM to generate rich semantic embeddings of user profiles (job descriptions, skills, bio), then use these as node features in the GNN.

```
"Senior ML Engineer at Google, worked on Search ranking,
 published paper on transformer efficiency..."
        │
        ▼
    [BERT / LLM encoder]
        │
        ▼
   profile_embedding (768-d)
        │
        ▼
   Input to GNN as node features
```

**LinkedIn's approach (published 2023):** Use a fine-tuned language model to encode user profiles, feed these into a GNN to produce connection recommendation embeddings.

**Pros:** Captures rich semantic similarity beyond just graph structure (two NLP researchers at different companies with similar papers get connected)

**Cons:** LLM inference at 1B users is expensive; typically done offline, embeddings cached

---

## Summary: Which to Use?

| Use case | Recommended approach |
|---|---|
| MVP / baseline | Common Neighbors + Adamic-Adar |
| Mid-size platform (<10M users) | Node2Vec + lightweight ranker |
| Large platform (>100M users) | GraphSAGE two-tower + ranking model |
| State-of-the-art (1B users) | PinSage-style GNN + LLM profile features |

**For the interview:** Start by proposing the two-tower with GNN approach, explain you'd baseline with heuristics first, and be ready to discuss the trade-offs of each level.

---

## Interview Checkpoint

**Q: Why can't you just use a standard neural network without the graph structure?**

A standard MLP treats each user independently. It misses the crucial relational signal: two users who have 20 mutual connections are far more likely to connect than two users with similar profile text but no graph overlap. GNNs explicitly encode this neighborhood structure.

**Q: What is the difference between transductive and inductive GNN learning?**

- **Transductive** (e.g., basic GCN): The model is trained on a fixed graph and cannot produce embeddings for nodes not seen during training. Problematic for a platform where new users join daily.
- **Inductive** (e.g., GraphSAGE): The model learns an aggregation function that can be applied to any node, including new ones. It generates embeddings by aggregating the new node's existing neighbors. This is essential for production systems.

**Q: How does PinSage differ from a basic GNN?**

PinSage replaces the fixed k-nearest-neighbor sampling with importance sampling via random walks, making the computation subgraph more focused on truly influential neighbors. It also uses a producer-consumer pipeline for throughput and curriculum learning for training stability.
