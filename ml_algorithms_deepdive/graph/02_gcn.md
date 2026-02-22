# Graph Convolutional Networks (GCN)

## The Core Idea

Graph Convolutional Networks (GCN) apply the idea of **convolution** — the same operation that makes CNNs work on images — to graph-structured data.

In a CNN, a filter slides over an image and aggregates neighboring pixels:

```
Image:
[ 1  2  3 ]
[ 4  5  6 ]   → convolve → new representation for pixel 5
[ 7  8  9 ]
(pixel 5's new value depends on pixels 2, 4, 5, 6, 8)
```

In a GCN, each node aggregates information from its neighbors:

```
Graph:
   B
   |
A — C — D     → aggregate → new representation for C
   |
   E
(C's new value depends on A, B, C, D, E)
```

Both operations answer the same question: **what does the local neighborhood look like?**

---

## Why This Works

The key insight: **a node's label is correlated with its neighbors' labels.**

In a fraud detection graph:
- A genuine user connected to 5 fraudsters should itself be suspicious
- A fraudster's embedding should "infect" the embeddings of its neighbors

By repeatedly aggregating neighbor information, a GCN can propagate signals across multiple hops.

---

## How GCN Works

### Step 1: Start with Node Features

Each node starts with a feature vector. These can be:
- User profile features (age, location, engagement rate)
- Item features (category, price, text embeddings)
- One-hot encodings if no features exist

```
Node A: [0.1, 0.5, 0.2]   (3-dim feature vector)
Node B: [0.3, 0.1, 0.8]
Node C: [0.9, 0.2, 0.4]
```

### Step 2: Message Passing (one layer)

For each node, collect the feature vectors of all its neighbors (including itself), average them, and apply a linear transformation + activation:

```
h_C^(1) = ReLU( W · mean([h_A, h_B, h_C, h_D, h_E]) )
```

Where:
- `h_C^(1)` = new embedding of node C after 1 layer
- `W` = learned weight matrix (same W for all nodes)
- `mean(...)` = average of neighbor features
- `ReLU` = non-linear activation

The weight matrix W is **shared across all nodes**. This is what makes GCN a "convolution" — the same filter applied everywhere.

### Step 3: Stack Multiple Layers

Each layer expands the receptive field by one hop:

```
After 1 layer: each node knows about its 1-hop neighbors
After 2 layers: each node knows about its 2-hop neighbors
After 3 layers: each node knows about its 3-hop neighbors
```

```
Layer 0 (input):        A    B    C    D    E
                         \   |   /|\   |   /
Layer 1 (1-hop):          B  C  A C D  C  D
                               \ | /
Layer 2 (2-hop):                A,B,C,D,E all mixed
```

### Step 4: Use the Final Embeddings

After k layers, each node has a k-hop-aware embedding. Use these for:
- **Node classification**: pass embeddings through a classifier (softmax)
- **Link prediction**: compute similarity between two node embeddings
- **Graph classification**: pool all node embeddings into one graph embedding

---

## The Math (Kipf & Welling, 2017)

The GCN update rule per layer:

```
H^(l+1) = σ( D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l) )
```

Where:
- `H^(l)` = node embeddings at layer l (matrix: N_nodes × d_features)
- `Ã = A + I` = adjacency matrix + self-loops (include the node itself in aggregation)
- `D̃` = degree matrix of Ã (used for normalization)
- `W^(l)` = learned weight matrix at layer l
- `σ` = activation function (ReLU)

The normalization `D̃^(-1/2) Ã D̃^(-1/2)` prevents high-degree nodes from dominating. A node with 1000 friends vs. 5 friends — without normalization, the hub node's signal would overwhelm everything.

---

## Worked Example: Fraud Detection

```
Graph:
  Genuine_1 —— Genuine_2
      |               |
  Suspicious —— Fraudster_1
      |
  Fraudster_2
```

Node features: [transaction_amount, account_age, num_failed_logins]

**Layer 1**: Suspicious's new embedding = mix of its own features + Genuine_1's + Fraudster_1's + Fraudster_2's features.

**Layer 2**: Genuine_1's new embedding = mix of itself + Genuine_2 + Suspicious (who now carries Fraudster signal from Layer 1).

After 2 layers: Genuine_1's embedding has been contaminated with fraudster signal from 2 hops away — even though it's directly connected only to Genuine_2 and Suspicious.

This is exactly the propagation we want for fraud detection.

---

## Training

GCN is trained end-to-end with standard backpropagation:

```
For node classification:
  Loss = CrossEntropy(softmax(W_classifier · h_v), true_label_v)

Gradients flow back through the aggregation layers.
```

You only need labels for **some** nodes (semi-supervised). The graph structure propagates supervision signal to unlabeled nodes through the aggregation.

---

## Trade-offs

### Strengths
- **Inductive within graph**: learns generalizable feature transformation, not just memorizing node IDs
- **Integrates node features**: unlike DeepWalk/Node2Vec, uses actual node attributes
- **End-to-end**: embeddings optimized directly for the downstream task
- **Scalable in depth**: expressiveness grows with number of layers

### Weaknesses

| Weakness | Explanation |
|----------|-------------|
| **Transductive** | Still requires all nodes to be present at training time to compute adjacency matrix |
| **Over-smoothing** | With many layers (>3-4), all node embeddings converge to similar values — "everyone looks the same" |
| **Scalability** | Full-batch training requires the entire graph in memory. Fails on million-node graphs |
| **Uniform neighbor weighting** | Treats all neighbors equally — a celebrity friend and a close friend get the same weight |
| **Shallow in practice** | 2-3 layers is usually optimal; more is worse due to over-smoothing |

---

## Over-smoothing: The Key Practical Limitation

```
Layer 0:  A=red, B=blue, C=green, D=yellow

Layer 1:  A≈mix(A,B,C), B≈mix(A,B,D), C≈mix(A,C,D), D≈mix(B,C,D)

Layer 10: A≈B≈C≈D≈gray  ← everything converges, useless
```

In practice: **2-3 layers is the sweet spot** for most graph ML tasks. This means GCN can only capture 2-3 hop neighborhoods effectively.

This motivated GraphSAGE, GAT, and other variants.

---

## When to Use GCN

| Use Case | GCN? |
|----------|------|
| Node classification on citation networks | ✅ Classic use case |
| Fraud detection on transaction graphs | ✅ Works well |
| Social network friend recommendation | ⚠️ Use GraphSAGE instead (scale) |
| New nodes join graph frequently | ❌ GCN is transductive |
| Billion-node graphs | ❌ Too slow |
| Need to weight different neighbors differently | ❌ Use GAT |

---

## Industry Examples

- **Alibaba**: GCN-based graph for e-commerce fraud detection
- **Academic benchmark**: Cora and Citeseer citation network node classification (GCN's original benchmark)

---

## Key Interview Points

1. **Why GCN and not a standard MLP?** → MLPs process each node independently; GCN lets nodes share information through the graph structure.

2. **What does each GCN layer do?** → One layer = one hop of neighborhood aggregation.

3. **Why not use 10 layers?** → Over-smoothing: all embeddings converge.

4. **How does GCN handle directed graphs?** → Use the directed adjacency matrix; aggregation only flows along edge direction.

5. **Transductive vs. inductive?** → GCN is transductive (needs all nodes at training time). GraphSAGE is inductive.

---

## Next

See `03_graphsage.md` — GraphSAGE, which fixes GCN's scalability and transductive limitations.
