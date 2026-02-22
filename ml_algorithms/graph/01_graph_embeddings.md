# Graph Embeddings: DeepWalk, Node2Vec, LINE

## The Core Idea

Before deep learning on graphs existed, researchers asked: **can we learn a vector for each node that captures its position and neighborhood in the graph?**

If node A and node B play similar structural roles (e.g., both are highly connected hubs with similar neighbors), their vectors should be close in embedding space. Then you can use those vectors as features in any standard ML model.

This is exactly what word embeddings (Word2Vec) do for words in a sentence — and graph embedding methods borrow that intuition directly.

---

## DeepWalk (2014)

### The Analogy: Words in a Sentence → Nodes in a Walk

In Word2Vec:
- A **sentence** is a sequence of words
- Words that appear near each other get similar embeddings

In DeepWalk:
- A **random walk** is a sequence of nodes
- Nodes that appear near each other in random walks get similar embeddings

### How It Works

**Step 1: Generate random walks**

Start at each node. At each step, randomly move to one of the current node's neighbors. Record the sequence.

```
Graph:  A — B — C — D
            |
            E — F

Starting at A, a random walk might produce:
Walk 1: A → B → C → D → C → B → E
Walk 2: A → B → E → F → E → B → C
Walk 3: B → E → B → A → B → C → D
```

Run many walks (e.g., 10 walks per node, each of length 80).

**Step 2: Train Skip-gram (Word2Vec) on the walks**

Treat each walk as a sentence. Train Word2Vec to predict, given a node, which nodes appear nearby in a walk.

```
Walk: [A, B, C, D, C, B, E]
       ↑
For center node C, predict neighbors within window: [A, B, D, B]
```

The training pushes embeddings of co-occurring nodes close together.

**Step 3: Use the embeddings**

Each node now has a d-dimensional vector (e.g., 128-dim). Use these as features for downstream tasks: node classification, link prediction, visualization.

### Trade-offs

| Aspect | Detail |
|--------|--------|
| **Simple** | Easy to implement, no labels needed |
| **Scalable** | Random walks are fast and parallelizable |
| **Transductive** | Cannot embed new nodes not seen during training |
| **No features** | Ignores node/edge attributes entirely |
| **Uniform neighbor sampling** | Doesn't distinguish between types of neighbors |

### Use Cases
- Social network community detection
- Biological network analysis
- Baseline for link prediction

---

## Node2Vec (2016)

### The Problem with DeepWalk

DeepWalk's random walks are purely random — equal probability of going to any neighbor. This doesn't let you control what kind of structure gets captured.

Two types of structure matter in graphs:

**Homophily**: nodes that are close (in terms of hops) should be similar.
> "People in the same friend group have similar interests."

**Structural equivalence**: nodes that play similar *roles* in the graph should be similar, even if far apart.
> "All hub nodes (high-degree nodes) are similar to each other, even if they're in different parts of the graph."

### The Solution: Biased Random Walks

Node2Vec introduces two parameters:
- **p** (return parameter): probability of going *back* to the previous node
- **q** (in-out parameter): probability of going *outward* vs. staying close

```
          prev_node
              |
        current_node
        /    |    \
    d=1    d=1    d=2
   (back)  (same) (away)
```

- **Low p**: walk comes back often → explores local neighborhood → captures **homophily**
- **Low q**: walk goes far away → explores distant nodes → captures **structural equivalence**

### Example

```
Graph: Two dense clusters connected by a bridge

Cluster 1:  A—B—C—D—E (dense)
                |
              Bridge
                |
Cluster 2:  F—G—H—I—J (dense)
```

- **DFS-like walk (low q)**: A→B→C→D→C→B→A — stays in Cluster 1, captures community structure
- **BFS-like walk (low p)**: A→B→C→G→H→I — crosses the bridge, explores the graph structure

### Trade-offs vs. DeepWalk

| Aspect | DeepWalk | Node2Vec |
|--------|----------|----------|
| Walk strategy | Uniform random | Biased (controlled by p, q) |
| Captures homophily | Partially | Yes (low q) |
| Captures structural equivalence | Weakly | Yes (low p) |
| Complexity | O(walks) | O(walks), slightly slower due to bias computation |
| Tuning required | No | Yes (p, q are hyperparameters) |

### Use Cases
- Friend recommendation (low q — homophily)
- Role detection in organizational networks (low p — structural equivalence)
- Protein function prediction

---

## LINE (Large-scale Information Network Embedding, 2015)

### The Problem It Solves

DeepWalk and Node2Vec work on small-to-medium graphs. LINE was designed for web-scale graphs with **billions of nodes and edges**.

### The Idea: First-order and Second-order Proximity

**First-order proximity**: nodes directly connected should be similar.
> If Alice and Bob are friends, their embeddings should be close.

**Second-order proximity**: nodes that share many common neighbors should be similar.
> If Alice and Carol both know Bob, Dave, and Eve, they're probably similar even if they've never met.

LINE trains two separate objectives:

**Objective 1 (first-order)**:
Minimize the difference between the empirical distribution of connected pairs and the model's distribution.

**Objective 2 (second-order)**:
Treat each node as both a "node" and a "context node." Nodes that appear in similar contexts (neighborhoods) should be similar.

### Why It Scales

- Processes edges one by one (doesn't need the whole graph in memory)
- Uses negative sampling (same trick as Word2Vec)
- Can handle directed, weighted, and undirected graphs

### Trade-offs

| Aspect | Detail |
|--------|--------|
| **Scalable** | Designed for billion-scale graphs |
| **Explicit objectives** | First/second-order proximity is principled |
| **No node features** | Still ignores attributes |
| **Transductive** | Same limitation as DeepWalk |
| **Two separate embeddings** | First and second order trained separately, then concatenated |

---

## Shared Limitations of All Graph Embedding Methods

These are the reasons Graph Neural Networks (GNNs) were developed:

1. **Transductive**: can only embed nodes seen during training. A new user added to the graph tomorrow gets no embedding.

2. **No feature integration**: node attributes (age, job title, text bio) are ignored. The embedding only captures graph structure.

3. **No end-to-end learning**: embeddings are pre-trained separately from the downstream task. You can't optimize the embeddings directly for, say, fraud detection.

4. **Fixed graph**: the graph must be static during training. Dynamic graphs (users joining, edges forming in real time) are hard to handle.

GNNs address all four of these. But embedding methods are still used:
- As fast baselines
- When the graph has no node features
- As pre-training for GNN initialization
- For ultra-large graphs where GNNs are too slow

---

## Quick Comparison

| Method | Year | Walk Type | Node Features | Inductive | Scale |
|--------|------|-----------|---------------|-----------|-------|
| DeepWalk | 2014 | Uniform random | No | No | Medium |
| Node2Vec | 2016 | Biased (p, q) | No | No | Medium |
| LINE | 2015 | N/A (edge-based) | No | No | Large |

---

## Industry Use

- **Pinterest**: used Node2Vec-style embeddings to initialize graph neural networks for recommendation (PinSage)
- **LinkedIn**: used graph embeddings for "People You May Know" before transitioning to GNNs
- **Twitter**: used graph embeddings for account recommendations and community detection

---

## Next

See `02_gcn.md` — Graph Convolutional Networks, the foundational deep learning approach that overcomes the limitations above.
