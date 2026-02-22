# Graph Attention Networks (GAT)

## The Problem: Not All Neighbors Are Equal

Both GCN and GraphSAGE treat neighbors with **equal weight** (up to degree normalization).

But in reality:
- Your closest friend's opinion matters more than a casual acquaintance's
- A paper cited 100 times matters more than one cited twice
- A hub node's influence should be discounted because it has thousands of connections

```
Bob
 |——— Alice (close friend, daily interaction)
 |——— Carol (met once at a conference)
 |——— Dave  (company connection)

GCN/SAGE: Alice = Carol = Dave in Bob's aggregation
GAT:      Alice gets weight 0.7, Carol 0.2, Dave 0.1
```

Graph Attention Networks (GAT, Veličković et al. 2018) solve this by **learning the importance of each neighbor dynamically**.

---

## The Core Idea: Attention Weights

GAT computes an **attention score** between every pair of connected nodes. These scores are learned during training and tell the model how much to weight each neighbor when aggregating.

```
Before aggregation:
  Neighbors of Bob: [Alice, Carol, Dave]
  Attention scores: [0.7,   0.2,   0.1]

Aggregation:
  h_Bob = 0.7 × h_Alice + 0.2 × h_Carol + 0.1 × h_Dave
```

Crucially, the attention weights are **data-dependent** — they change based on the features of the nodes involved. Two different target nodes can assign completely different weights to the same neighbor.

---

## How GAT Computes Attention

### Step 1: Linear Transformation

Apply a shared weight matrix W to all node features:

```
z_v = W · h_v    (for all nodes v)
```

### Step 2: Compute Raw Attention Score

For each edge (i, j), compute how much node j should attend to node i:

```
e_ij = LeakyReLU( a^T · concat(z_i, z_j) )
```

Where:
- `a` = learned attention vector
- `concat(z_i, z_j)` = concatenation of the two node embeddings
- `LeakyReLU` = prevents dying gradients

Intuition: `a^T · concat(z_i, z_j)` is a learned compatibility function — it scores how relevant node i is to node j.

### Step 3: Normalize with Softmax

Normalize scores across all of node j's neighbors so they sum to 1:

```
α_ij = softmax_i( e_ij ) = exp(e_ij) / Σ_k exp(e_kj)
```

These are the final attention weights.

### Step 4: Weighted Aggregation

```
h_j^(new) = σ( Σ_i  α_ij · z_i )
```

Node j's new embedding = weighted sum of neighbor embeddings, weighted by learned attention.

---

## Multi-Head Attention

GAT uses **multi-head attention** (same idea as Transformers).

Run K independent attention mechanisms in parallel, then concatenate (or average) the results:

```
h_j^(new) = concat( σ(Σ_i α_ij^(1) z_i^(1)),
                     σ(Σ_i α_ij^(2) z_i^(2)),
                     ...,
                     σ(Σ_i α_ij^(K) z_i^(K)) )
```

Why? Different attention heads can learn different aspects of neighbor importance:
- Head 1: attends to neighbors with similar features
- Head 2: attends to high-degree neighbors (hubs)
- Head 3: attends to neighbors with different labels (for boundary nodes)

Typical: K = 8 heads for intermediate layers, averaged (not concatenated) at the last layer.

---

## Attention as Interpretability

A big advantage of GAT: the attention weights are human-interpretable.

Example: in a drug interaction graph, you can visualize which protein interactions the model focused on for a specific drug's prediction. This is valuable for scientific understanding and debugging.

```
Drug A prediction: toxic
  Protein_X attention: 0.82  ← model strongly focused here
  Protein_Y attention: 0.12
  Protein_Z attention: 0.06

Biologist: "Protein_X is the mechanism — makes sense."
```

---

## GAT vs. GCN vs. GraphSAGE

| Aspect | GCN | GraphSAGE | GAT |
|--------|-----|-----------|-----|
| Neighbor weighting | Fixed (degree-normalized) | Equal (among sampled) | Learned (attention) |
| Expressiveness | Low | Medium | High |
| Inductive | No | Yes | Yes |
| Scalability | Medium | Large | Medium (attention is O(edges)) |
| Interpretability | Low | Low | High (attention weights) |
| Multi-head support | No | No | Yes |
| Complexity | O(N·d²) | O(batch·fanout·d²) | O(E·d) where E=num edges |

---

## Practical Considerations

### When Attention Helps
- **Heterogeneous neighborhoods**: some neighbors are very relevant, others are noise
- **Noisy graphs**: real-world graphs have many spurious edges; attention can down-weight them
- **Long-range dependencies**: combined with few layers, attention can amplify distant but important signals

### When Attention Doesn't Help
- **Homogeneous, dense graphs**: if all neighbors are equally relevant, attention adds overhead without benefit
- **Very large graphs**: computing attention for all edges is O(E × d), expensive at billion-edge scale

### Scalability Concern

GAT computes attention for every edge. If a graph has 10 billion edges:
- Attention computation = 10B × d operations
- Memory = 10B attention scores

Compare to GraphSAGE: samples ~25 neighbors per node, far fewer computations.

**Solution: GraphSAGE + attention**. Sample neighbors with GraphSAGE, then apply attention only to the sampled set. This is what many production systems do.

---

## GAT vs. Transformer: The Connection

GAT is essentially a **Transformer applied to graphs**:

| Transformer | GAT |
|-------------|-----|
| Attention over sequence positions | Attention over graph neighbors |
| Queries, Keys, Values | Derived from node features |
| All positions attend to all others | Only connected nodes attend to each other |
| Positional encoding | Graph structure provides the "position" |

In fact, modern graph transformers (Graphormer, GPS) extend this further by allowing *all* nodes to attend to *all* others, not just neighbors.

---

## Use Cases

| Use Case | Why GAT? |
|----------|----------|
| Citation network node classification | Weights relevant citations higher |
| Knowledge graph reasoning | Different edge types need different attention |
| Drug-drug interaction prediction | Need to identify which interactions matter most |
| Protein structure prediction (AlphaFold-like) | Residue pairs at contact have high attention |
| Recommendation with heterogeneous behavior | "Clicked" edges vs. "purchased" edges need different weights |
| Fraud detection | Fraudster accounts have structurally distinct neighborhood patterns |

---

## Industry Examples

- **Alibaba**: GAT used in product recommendation graphs where different user interaction types (click, cart, purchase) need different weights
- **Biology**: Graph-based drug discovery widely uses GAT for molecular property prediction
- **Microsoft**: GAT used in knowledge graph completion for question answering

---

## Key Interview Points

1. **How does GAT differ from GCN?**
   - GCN: fixed degree-based normalization. GAT: learned, data-dependent attention weights.

2. **What is multi-head attention in GAT?**
   - Run K independent attentions, concatenate results. Different heads capture different aspects of relevance.

3. **What are attention weights interpretable as?**
   - How much each neighbor contributed to a node's embedding. Useful for debugging and scientific insight.

4. **Scalability concern?**
   - O(E) attention computation. For billion-edge graphs, combine with GraphSAGE-style neighbor sampling first.

5. **When would you choose GAT over GraphSAGE?**
   - When neighbor importance is heterogeneous and interpretability matters. For pure scale, GraphSAGE wins.

---

## Next

See `05_advanced_gnns.md` — GIN (most theoretically expressive GNN), Heterogeneous GNNs for multi-type graphs (HGT, R-GCN), and Knowledge Graph Embeddings.
