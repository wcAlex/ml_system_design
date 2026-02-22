# Graph ML: Summary, Comparison, and Interview Decision Framework

## The Full Landscape at a Glance

```
Graph ML Algorithms
│
├── Graph Embeddings (pre-GNN era)
│   ├── DeepWalk       — random walks + Word2Vec
│   ├── Node2Vec       — biased random walks (controls homophily vs. structural equivalence)
│   └── LINE           — first/second-order proximity, billion-scale
│
├── Graph Neural Networks
│   ├── GCN            — foundational, spectral convolution
│   ├── GraphSAGE      — inductive, scalable, mini-batch
│   ├── GAT            — learned attention weights, interpretable
│   └── GIN            — maximally expressive (sum aggregation + MLP)
│
└── Specialized
    ├── R-GCN          — heterogeneous graphs (type-specific weight matrices)
    ├── HGT            — heterogeneous graphs with full attention (Transformer-style)
    ├── TransE         — knowledge graph completion (translation)
    └── RotatE         — knowledge graph completion (rotation, more expressive)
```

---

## Full Comparison Table

| Algorithm | Year | Inductive | Scale | Node Features | Hetero Graph | Aggregation | Best Task |
|-----------|------|-----------|-------|---------------|--------------|-------------|-----------|
| DeepWalk | 2014 | No | Medium | No | No | N/A (walk-based) | Node classification, community detection |
| Node2Vec | 2016 | No | Medium | No | No | N/A (walk-based) | Role detection, link prediction |
| LINE | 2015 | No | Billion | No | No | N/A (edge-based) | Large-scale embedding |
| GCN | 2017 | No | Medium | Yes | No | Mean (normalized) | Node classification |
| GraphSAGE | 2017 | Yes | Billion | Yes | No | Mean/LSTM/Max-pool | Link prediction, new-node embedding |
| GAT | 2018 | Yes | Medium | Yes | No | Attention-weighted | Node classification, interpretable tasks |
| GIN | 2019 | Yes | Medium | Yes | No | Sum + MLP | Graph classification |
| R-GCN | 2018 | No | Medium | Yes | Yes | Per-relation mean | KG completion, relational graphs |
| HGT | 2020 | Yes | Medium | Yes | Yes | Type-aware attention | Heterogeneous recommendation |
| TransE | 2013 | No | Billion | No | Yes | N/A (triple-based) | Knowledge graph completion |
| RotatE | 2019 | No | Billion | No | Yes | N/A (triple-based) | KG completion (complex patterns) |

---

## Decision Framework for Interviews

Use this tree to pick the right algorithm during an interview:

```
Is the graph heterogeneous (multiple node/edge types)?
│
├── Yes ──→ Is it a knowledge graph with factual triples?
│           │
│           ├── Yes, simple relations ──→ TransE
│           ├── Yes, complex patterns ──→ RotatE
│           └── No (social/e-commerce) ──→ HGT or R-GCN
│
└── No ──→ Do you need to handle new nodes at inference?
           │
           ├── No ──→ Is the graph small (<10M nodes)?
           │          │
           │          ├── Yes ──→ GCN (simple) or GAT (if neighbor importance varies)
           │          └── No  ──→ Graph Embeddings (DeepWalk/Node2Vec) as baseline
           │
           └── Yes ──→ Is scale >100M nodes?
                       │
                       ├── Yes ──→ GraphSAGE (most scalable inductive GNN)
                       └── No  ──→ Do neighbors have varying importance?
                                   │
                                   ├── Yes ──→ GAT
                                   └── No  ──→ GraphSAGE
```

---

## Mapping to Real Systems

### LinkedIn — "People You May Know"

```
Graph: User — WorksAt → Company
       User — HasSkill → Skill
       User — Knows → User
       User — StudiedAt → School

Approach:
  1. HGT or R-GCN for heterogeneous graph
  2. Task: link prediction (User-User edges)
  3. Node features: profile data, activity features
  4. Scale: GraphSAGE-style mini-batch for billion-scale

LinkedIn published: "Graph Neural Networks for Social Recommendations" (2021)
```

### Pinterest — Image Recommendation (PinSage)

```
Graph: Pin — SavedBy → Board
       Pin — SimilarTo → Pin (visual similarity)

Approach:
  1. GraphSAGE (inductive — new pins added every day)
  2. Task: link prediction (pin-pin similarity)
  3. Node features: visual embeddings from CNN
  4. Random walk-based neighbor sampling for scalability
  5. Hard negative mining for better training

Scale: 3 billion pins, 18 billion edges
```

### Uber Eats — Food Recommendation

```
Graph: User — Ordered → MenuItem
       MenuItem — BelongsTo → Restaurant
       User — LivesIn → Location

Approach:
  1. GraphSAGE on user-item bipartite graph
  2. Task: link prediction (user-item)
  3. Node features: user history, item metadata
  4. Incorporates geographic proximity as edge feature

Published: "Food Discovery with Uber Eats" (2019)
```

### Alibaba — E-commerce Recommendation

```
Graph: User — Clicked → Item
       User — Purchased → Item
       Item — InCategory → Category

Approach:
  1. GAT (attention distinguishes "clicked" vs "purchased" behavior)
  2. GNN deployed in Taobao for real-time recommendation
  3. Mini-batch GraphSAGE at billion scale

Published: "Graph Convolutional Neural Networks for Web-Scale Recommender Systems" (2018)
```

### Google — Knowledge Graph

```
Graph: Entity — Relation → Entity (factual triples)

Approach:
  1. TransE / RotatE for KG completion
  2. Also uses GNN-based approaches for multi-hop reasoning
  3. Powers search knowledge panels and Q&A
```

---

## Common Interview Scenarios and Expected Answers

### "Design a friend recommendation system like LinkedIn's PYMK"

**Structure**:
1. Frame as link prediction on a user-user graph
2. Graph is heterogeneous (users, companies, skills, schools)
3. Choose HGT or R-GCN for heterogeneous message passing
4. Training: positive edges = existing connections, negative = random non-connections
5. Scale: GraphSAGE-style mini-batch, ~25-10 fanout per layer
6. Serving: offline embed all users weekly, ANN lookup at query time
7. Cold start: new user gets embedding from profile features alone (inductive)

**Key talking points**:
- Why not matrix factorization? → doesn't use rich node features, not inductive
- Why not collaborative filtering? → same issue, plus graph structure gives structural signals (friends of friends)
- Bottleneck: billion-scale graph training — use cluster-GCN or mini-batch GraphSAGE
- Challenge: graph changes constantly (new connections) — need inductive model

### "How does over-smoothing affect your GNN design?"

- With many layers, all node embeddings converge to the same vector
- In practice: use 2-3 layers maximum
- Workarounds: residual connections (add original embedding back), graph dropout, jumping knowledge networks (aggregate outputs from all layers)
- Alternative: increase expressiveness with attention (GAT) rather than depth

### "How do you handle cold start in a graph model?"

- **New node (no edges)**: use the node's feature vector directly; an inductive model (GraphSAGE, GAT) can embed it
- **New node (some edges)**: after a few interactions, sample its new neighbors and embed normally
- **Zero features + zero edges**: fall back to content-based model or popularity-based

### "How do you train a GNN at billion-node scale?"

1. **Neighbor sampling (GraphSAGE)**: sample K neighbors per node per layer, mini-batch gradient descent
2. **Cluster-GCN**: partition the graph into clusters, train within clusters
3. **GraphSAINT**: sample subgraphs (node-wise, edge-wise, or random walk), train on subgraphs
4. **Offline pre-computation**: pre-compute multi-hop neighbor aggregations, train a standard MLP on cached features (fast, loses end-to-end but very scalable)

---

## Key Terminology Cheatsheet

| Term | Definition |
|------|------------|
| **Transductive** | Model only works on nodes seen during training |
| **Inductive** | Model can embed new, unseen nodes |
| **Message passing** | Each node collects information from neighbors |
| **Aggregation** | How neighbor messages are combined (mean, sum, max, attention) |
| **Receptive field** | How many hops of neighbors a node can see (= number of layers) |
| **Over-smoothing** | Embeddings converge with many layers; all nodes look the same |
| **Homophily** | Connected nodes tend to have similar labels (common in social networks) |
| **Heterophily** | Connected nodes tend to have different labels (e.g., bipartite buyer-seller graphs) |
| **Link prediction** | Predict whether an edge should exist between two nodes |
| **Node classification** | Predict a label for each node |
| **Graph classification** | Predict a label for an entire graph |
| **ANN index** | Approximate nearest neighbors (FAISS, ScaNN) — fast lookup for GNN embeddings at serving time |
| **Fanout** | Number of neighbors sampled per node per layer in GraphSAGE |
| **WL test** | Weisfeiler-Lehman isomorphism test — theoretical upper bound on GNN expressiveness |
