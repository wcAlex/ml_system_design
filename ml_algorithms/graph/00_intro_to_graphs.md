# Introduction to Graphs in Machine Learning

## What is a Graph?

A graph is a data structure made of two things:
- **Nodes** (also called vertices): the entities
- **Edges**: the relationships between entities

```
   Alice ——— Bob
     |         |
   Carol ——— Dave
```

In the diagram above:
- Nodes = Alice, Bob, Carol, Dave (people)
- Edges = friendships between them

That's it. A graph is just a way to represent **things and their connections**.

---

## Why Graphs?

Most real-world data is relational, not tabular. Traditional ML works great on rows of independent features. But what if the features of one entity depend on who they are *connected* to?

| Problem | Why a table fails | Why a graph helps |
|---------|-------------------|-------------------|
| Friend recommendation | Users aren't independent — friends of friends matter | Explicit structure captures paths and neighborhoods |
| Fraud detection | One fraudster connects to many victims | Fraud often forms detectable patterns in the graph |
| Drug interaction | Drugs interact with proteins and other drugs | The interaction network is the data |
| Product recommendation | Bought-together relationships between items | Graph structure captures co-purchase patterns |
| Search ranking | Web pages link to each other (PageRank) | Link structure signals authority |

---

## Graph Terminology

```
     A
    / \
   B   C
   |   |
   D   E
    \ /
     F
```

| Term | Meaning | Example above |
|------|---------|---------------|
| **Node** | An entity | A, B, C, D, E, F |
| **Edge** | A connection between two nodes | A-B, A-C, B-D, ... |
| **Neighbor** | Nodes directly connected | Neighbors of A = {B, C} |
| **Degree** | Number of edges a node has | Degree of A = 2 |
| **Path** | Sequence of connected nodes | A → B → D → F |
| **Shortest path** | Minimum hops between two nodes | A to F: A→B→D→F (3 hops) |
| **Subgraph** | A portion of the graph | |
| **Connected component** | A cluster with no edges to the rest | |

---

## Types of Graphs

### Undirected vs. Directed

**Undirected**: edges have no direction. Friendship is mutual.
```
Alice ——— Bob   (Alice is friends with Bob, Bob is friends with Alice)
```

**Directed**: edges have direction. Following on Twitter is one-way.
```
Alice ——→ Bob   (Alice follows Bob, but Bob doesn't follow Alice)
```

### Weighted vs. Unweighted

**Unweighted**: all edges are equal.
```
Alice ——— Bob ——— Carol
```

**Weighted**: edges have a strength or cost.
```
Alice —(0.9)— Bob —(0.2)— Carol
(Strong friends)     (Weak friends)
```

### Homogeneous vs. Heterogeneous

**Homogeneous**: one type of node and one type of edge.
- Social network: all nodes are people, all edges are friendships

**Heterogeneous**: multiple types of nodes and/or edges.
- LinkedIn: nodes are [Users, Companies, Skills], edges are [WorksAt, HasSkill, Knows]

---

## How Graphs Are Stored in Code

### Adjacency Matrix
A matrix where entry `[i][j] = 1` if there's an edge from node i to node j.

```
     A  B  C  D
A  [ 0  1  1  0 ]
B  [ 1  0  0  1 ]
C  [ 1  0  0  1 ]
D  [ 0  1  1  0 ]
```

- **Pro**: O(1) edge lookup
- **Con**: O(N²) memory — terrible for large sparse graphs (most real graphs)

### Adjacency List
Each node stores a list of its neighbors.

```python
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'D'],
    'D': ['B', 'C'],
}
```

- **Pro**: Memory-efficient for sparse graphs
- **Con**: O(degree) edge lookup

### Edge List
A flat list of (source, destination) pairs.

```python
edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')]
```

- Most common format for loading into graph ML frameworks (PyG, DGL)

---

## The Core ML Problem on Graphs

Once we have a graph, we typically want to do one of three tasks:

### 1. Node Classification
Predict a label for each node.

> Example: Given a social network, classify each user as "bot" or "human."

```
Alice (?) —— Bob (bot) —— Carol (?)
              |
            Dave (bot)
```
Alice and Carol are connected to known bots. The graph structure is evidence they might also be bots.

### 2. Link Prediction
Predict whether an edge *should* exist between two nodes.

> Example: "People you may know" on LinkedIn. Should there be an edge between User A and User B?

```
Alice ——— Bob ——— Carol
              ?
Alice ——————————— Carol  ← should this edge exist?
```

### 3. Graph Classification
Predict a label for an entire graph.

> Example: Given the molecular structure of a drug (graph of atoms and bonds), predict whether it's toxic.

---

## Why Traditional ML Fails on Graphs

Standard ML assumes features are fixed-size vectors and samples are independent. Graphs break both assumptions:

1. **Variable neighborhood size**: Node A might have 2 neighbors, Node B might have 2000. You can't just concatenate neighbors into a fixed-size feature vector.

2. **Permutation invariance**: The order in which you list neighbors shouldn't matter. A model must be invariant to node ordering.

3. **Dependencies between nodes**: Nodes are not independent. Predicting one node's label depends on its neighbors, which depend on *their* neighbors, and so on.

Graph ML methods solve these problems by designing algorithms that naturally operate on graph structure.

---

## The Landscape of Graph ML Algorithms

```
Graph ML
├── Graph Embeddings (unsupervised, pre-deep-learning era)
│   ├── DeepWalk
│   ├── Node2Vec
│   └── LINE
│
├── Graph Neural Networks (deep learning on graphs)
│   ├── GCN  — Graph Convolutional Network (foundational)
│   ├── GraphSAGE — scalable, inductive
│   ├── GAT  — Graph Attention Network (learned neighbor weights)
│   └── GIN  — Graph Isomorphism Network (most expressive)
│
└── Specialized
    ├── Heterogeneous GNNs (HGT, R-GCN) — multiple node/edge types
    └── Knowledge Graph Embeddings (TransE, RotatE) — knowledge graphs
```

Each is covered in its own document. Start with `01_graph_embeddings.md`.
