# Advanced GNNs: GIN, Heterogeneous Graphs, and Knowledge Graph Embeddings

---

## Part 1: GIN — Graph Isomorphism Network

### The Expressiveness Problem

GCN, GraphSAGE, and GAT all use **sum or mean aggregation**. This creates a fundamental blind spot: they can confuse graphs that are structurally different.

### The Weisfeiler-Lehman Test

The WL graph isomorphism test is a classical algorithm to check if two graphs are identical. It works by repeatedly hashing a node's label with its neighbors' labels.

**Key theorem (Xu et al. 2019)**: Any GNN that uses sum/mean/max aggregation is **at most as powerful** as the WL test. Specifically:

- **Sum aggregation** = as powerful as WL (can distinguish all graphs WL can distinguish)
- **Mean aggregation** = weaker than WL (confuses graphs with different sizes but same proportions)
- **Max aggregation** = weaker than WL (loses count information)

GCN and GraphSAGE use mean, so they have limited expressiveness.

### What GIN Does Differently

GIN (Graph Isomorphism Network) uses **sum aggregation** with a learnable scalar ε:

```
h_v^(k+1) = MLP( (1 + ε) · h_v^(k)  +  Σ_{u ∈ N(v)} h_u^(k) )
```

Where:
- `ε` = learnable scalar (or fixed to 0) — how much to weight the node itself vs. its neighbors
- `MLP` = multi-layer perceptron (more powerful than a single linear layer)
- Sum (not mean) preserves count information

### Why Sum Matters: An Example

```
Graph A:                 Graph B:
  1 - 2 - 3               1 - 2 - 3
                               |
                               4

Node 2's neighbors in A: {1, 3}   → sum = 1 + 3 = 4
Node 2's neighbors in B: {1, 3, 4} → sum = 1 + 3 + 4 = 8   (different!)

Mean aggregation:
A: mean(1, 3) = 2
B: mean(1, 3, 4) = 2.67  (different, but less distinct)

Max aggregation:
A: max(1, 3) = 3
B: max(1, 3, 4) = 4  (different in this case, but other examples fail)
```

With sum: node 2 knows it has 2 vs. 3 neighbors, and their exact identities. Mean loses the count. Max loses most of the identity.

### GIN Trade-offs

| Aspect | Detail |
|--------|--------|
| **Most expressive** | Theoretically proven — as powerful as WL test |
| **Graph classification** | Excels here — needs to distinguish whole graph structures |
| **Node classification** | Often overkill — GCN/GAT sufficient |
| **Computational cost** | MLP at every layer is more expensive than a linear layer |
| **Not beyond WL** | Still cannot distinguish all non-isomorphic graphs (some remain indistinguishable) |

### When to Use GIN

- **Graph classification** (predicting properties of entire molecules, social networks)
- When you need maximum expressiveness within the standard GNN framework
- Chemistry/biology: predicting molecular properties where structure counts matter

---

## Part 2: Heterogeneous Graph Neural Networks

### What is a Heterogeneous Graph?

Real-world graphs have **multiple types of nodes and edges**:

```
LinkedIn graph:
  Nodes: [User, Company, Skill, School]
  Edges: [WorksAt(User→Company), HasSkill(User→Skill),
          StudiedAt(User→School), Knows(User→User)]
```

A standard GNN treats all nodes and edges identically. But "WorksAt" and "Knows" are fundamentally different relationships — they shouldn't be aggregated the same way.

### Problem with Standard GNNs on Heterogeneous Graphs

```
User A's neighbors:
  - Bob (User, connected via "Knows")
  - Google (Company, connected via "WorksAt")
  - Python (Skill, connected via "HasSkill")

GCN: aggregates all three with the same weight matrix W
     → mixes semantically different information
```

This is like treating apples and oranges as the same thing.

### R-GCN: Relational Graph Convolutional Network (2018)

The simplest fix: **use a separate weight matrix for each edge type**.

```
h_v^(new) = σ( Σ_r  Σ_{u ∈ N_r(v)}  (1/|N_r(v)|) W_r h_u  +  W_0 h_v )
```

Where:
- `r` = relation type (edge type)
- `W_r` = weight matrix specific to relation r
- `N_r(v)` = neighbors of v connected via relation r

Example for User A:
```
h_A^(new) = σ(
  W_Knows · mean(Bob's features)  +
  W_WorksAt · mean(Google's features)  +
  W_HasSkill · mean(Python's features)  +
  W_0 · h_A
)
```

**Problem**: if you have 1000 relation types, you need 1000 weight matrices. This explodes memory and overfits.

**Solution**: basis decomposition — represent W_r as a linear combination of K shared basis matrices:
```
W_r = Σ_k a_rk · B_k
```

This reduces parameters from (num_relations × d × d) to (K × d × d + num_relations × K).

### HGT: Heterogeneous Graph Transformer (2020)

HGT (from Microsoft Research) extends the GAT idea to heterogeneous graphs, using **type-specific attention**.

```
For edge (source_node, relation, target_node):
  - Use source_type-specific linear transform: W^{src_type}
  - Use target_type-specific linear transform: W^{tgt_type}
  - Use relation-specific attention: W^{relation}
  - Compute attention score based on all three
```

Architecture:
```
Attention(s, e, t) =
  softmax( (K(s, φ(s)) · Q(t, φ(t)) · W^{ATT}_{ψ(e)}) / √d )

Message(s, e, t) = M(s, φ(s)) · W^{MSG}_{ψ(e)}

h_t^{new} = Agg( {Attention(s,e,t) · Message(s,e,t) for all (s,e,t) ∈ neighbors} )
```

Where φ(v) = node type, ψ(e) = edge type.

HGT benefits:
- Full attention over heterogeneous graphs
- Type-aware transformations
- Handles dynamic graphs (time-aware variant)

### When to Use Heterogeneous GNNs

| Use Case | Method |
|----------|--------|
| Knowledge graph completion | R-GCN |
| LinkedIn people recommendation | HGT (user-user-company-skill graph) |
| E-commerce: user-item-category graph | R-GCN or HGT |
| Academic: paper-author-venue graph | HGT |
| Drug discovery: drug-protein-disease | HGT |

---

## Part 3: Knowledge Graph Embeddings

### What is a Knowledge Graph?

A knowledge graph stores **factual relationships** as triples:

```
(subject, relation, object)

Examples:
(Barack Obama, born_in, Hawaii)
(Apple, makes, iPhone)
(Aspirin, treats, Headache)
(Paris, capital_of, France)
```

Knowledge graphs are heterogeneous graphs where:
- Nodes = entities (people, places, things)
- Edges = typed relations

Google's Knowledge Graph, Freebase, Wikidata, and medical ontologies are all knowledge graphs.

### The Task: Knowledge Graph Completion

Many triples are missing. The task is to predict missing links:

```
(Steve Jobs, founded, ?)  → Apple ← should be inferred
(?, born_in, Seattle)     → Jimi Hendrix ← should be inferred
```

### TransE (2013): The Foundational Model

**Idea**: model a relation as a **translation** in embedding space.

If (h, r, t) is a true triple, then:
```
h + r ≈ t
```

Where:
- h = head entity embedding
- r = relation embedding
- t = tail entity embedding

```
Paris + capital_of ≈ France
→ embedding(Paris) + embedding(capital_of) ≈ embedding(France)
```

Visually:
```
Embedding space:
  Paris ──(capital_of)──→ France
  Berlin ──(capital_of)──→ Germany
  Tokyo ──(capital_of)──→ Japan

The "capital_of" vector consistently points from capital to country.
```

**Training**: minimize ||h + r - t|| for true triples, maximize it for false triples.

**Limitation**: TransE struggles with:
- 1-to-many relations: (person, speaks_language, ?) — one person speaks many languages
- Symmetric relations: (A, married_to, B) → (B, married_to, A) but h + r ≠ h

### RotatE (2019): Rotation in Complex Space

Instead of translation, model each relation as a **rotation** in complex number space:

```
t = h ∘ r   (element-wise product in complex space)
```

Where h and t are complex vectors, and r has unit magnitude (r = e^{iθ} — a pure rotation).

This handles:
- **Symmetry**: r ∘ r = 1 (rotate and rotate back = identity) → symmetric relations
- **Antisymmetry**: r ≠ r^{-1}
- **Inversion**: r^{-1} exists
- **Composition**: r1 ∘ r2 (first apply r1, then r2)

RotatE is state-of-the-art on most KG benchmarks (FB15k-237, WN18RR).

### Comparison: TransE vs. RotatE vs. GNN-based

| Method | Expressiveness | Symmetric | Antisymmetric | Composition | Scale |
|--------|---------------|-----------|----------------|-------------|-------|
| TransE | Low | ❌ | ✅ | ✅ | Large |
| RotatE | Medium | ✅ | ✅ | ✅ | Large |
| R-GCN | High | ✅ | ✅ | ✅ | Medium |
| HGT | Very High | ✅ | ✅ | ✅ | Medium |

KG embedding methods (TransE, RotatE) are simpler and faster but less expressive. GNN-based methods are more expressive but harder to scale.

### Industry Applications

- **Google**: Knowledge Graph powers the search knowledge panel
- **Meta**: Social graph reasoning for friend recommendations
- **Amazon**: Product-category-attribute graph for search and recommendation
- **Drug discovery**: BioKG (biological knowledge graph) for drug-target interaction prediction

---

## Summary: When to Use Which Method

| Algorithm | Best For | Key Advantage | Key Weakness |
|-----------|----------|---------------|--------------|
| GCN | Small static graphs, semi-supervised | Simple, well-understood | Transductive, doesn't scale |
| GraphSAGE | Large graphs, new node inference | Inductive, scalable | Equal neighbor weights |
| GAT | Heterogeneous neighborhoods, interpretability | Learned attention | Slower on huge graphs |
| GIN | Graph-level prediction, molecular ML | Maximally expressive | Overkill for node tasks |
| R-GCN | Heterogeneous graphs (few edge types) | Simple extension of GCN | Many parameters for many relations |
| HGT | Complex heterogeneous graphs (LinkedIn, KG) | Full type-aware attention | Complex to implement |
| TransE | Knowledge graph completion, simple facts | Fast, scalable | Can't handle symmetric relations |
| RotatE | Knowledge graph completion, complex patterns | Handles all relation patterns | Slower than TransE |

---

## Next

See `06_summary_and_comparison.md` — a full comparison table, interview decision framework, and mapping of algorithms to real-world systems.
