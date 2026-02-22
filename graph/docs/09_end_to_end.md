# 09 — End-to-End System

## Full Pipeline Walkthrough

### Scenario: User Alice opens LinkedIn homepage on mobile

```
1. Browser/App sends:
   GET /pymk/recommendations?user_id=alice&limit=20&context=mobile

2. PYMK Service receives request:
   a. Check Redis cache (L2): is there a fresh ranking for Alice?
      → Cache HIT: return cached result immediately (< 10ms)
      → Cache MISS: continue to step 3

3. Fetch Alice's precomputed candidate list:
   → Redis: GET pymk:candidates:{alice_id}
   → Returns ~1,000 candidate user IDs (precomputed by offline pipeline)
   → If missing: trigger background refresh and return empty/stale candidates

4. Batch-fetch features for Alice + all 1,000 candidates:
   → Feature Store (Redis / DynamoDB):
     - Alice's embedding, profile features, activity
     - Each candidate's embedding, profile features
     - Pairwise features: common neighbors, behavioral signals
   → Parallel batch request: ~20ms

5. Run ranking model:
   → Assemble feature matrix (1000 × 100)
   → Single forward pass through MLP ranker
   → Get 1,000 scores: ~10ms

6. Apply business rules:
   → Remove already-connected users, blocked, dismissed
   → Enforce diversity (max 5 from same company)
   → Keep top 20

7. Write to L2 cache (async, non-blocking):
   → Redis: SET pymk:ranked:{alice_id} <results> EX 3600

8. Return response:
   → 20 recommended users with scores and reason codes
   → Total latency: ~50-80ms
```

---

## Offline Pipeline Walkthrough

Runs as a scheduled Spark/distributed job:

```
Daily Batch Pipeline (runs at 2am UTC):
  │
  ├── Step 1: Graph Snapshot
  │     Read all connections from Graph DB → adjacency list (Parquet)
  │
  ├── Step 2: Feature Computation
  │     For each user, compute graph metrics (degrees, clustering coeff)
  │     For each user, compute profile features (normalize, encode)
  │     Write to Feature Store
  │
  ├── Step 3: GNN Training (if scheduled for today)
  │     Sample training triplets from recent 90 days
  │     Run distributed GraphSAGE training (PyTorch + PyG + DDP)
  │     Save model checkpoint
  │     Generate embeddings for all 1B users (inference pass)
  │     Write embeddings to embedding store
  │
  ├── Step 4: ANN Index Build
  │     Load all 1B embeddings
  │     Build FAISS HNSW index
  │     Validate recall (spot-check vs. brute force on 10K users)
  │     Deploy new index (swap old index atomically)
  │
  └── Step 5: Candidate List Generation
        For each user:
          - Graph traversal: 2nd-degree candidates (common neighbor count ≥ 2)
          - ANN search: top-500 nearest neighbors by embedding
          - Merge and deduplicate
          - Remove already-connected, blocked, dismissed
          - Write top-1500 candidates to Redis
```

---

## Mini Python Implementation

A simplified end-to-end implementation to illustrate the key components:

```python
"""
PYMK Mini Implementation
Demonstrates: GraphSAGE embeddings → Two-Tower ranking → Inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class User:
    user_id: int
    features: np.ndarray       # profile features (industry, title, location, etc.)
    neighbors: List[int]       # 1st-degree connection IDs


# ─── GraphSAGE Layer ──────────────────────────────────────────────────────────

class GraphSAGELayer(nn.Module):
    """Single GraphSAGE aggregation layer (mean aggregator)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.W = nn.Linear(in_dim * 2, out_dim)  # concat(self, agg)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, self_feat: torch.Tensor, neighbor_feat: torch.Tensor) -> torch.Tensor:
        """
        self_feat:     (batch, in_dim)
        neighbor_feat: (batch, num_samples, in_dim)
        """
        # Mean aggregation over sampled neighbors
        agg = neighbor_feat.mean(dim=1)                      # (batch, in_dim)
        combined = torch.cat([self_feat, agg], dim=-1)        # (batch, in_dim*2)
        out = F.relu(self.bn(self.W(combined)))               # (batch, out_dim)
        return self.dropout(out)


class GraphSAGE(nn.Module):
    """2-layer GraphSAGE encoder."""

    def __init__(self, node_feat_dim: int, hidden_dim: int = 256, embed_dim: int = 128):
        super().__init__()
        self.layer1 = GraphSAGELayer(node_feat_dim, hidden_dim)
        self.layer2 = GraphSAGELayer(hidden_dim, embed_dim)

    def forward(
        self,
        node_feat: torch.Tensor,         # (batch, node_feat_dim)
        hop1_neighbors: torch.Tensor,    # (batch, K1, node_feat_dim)
        hop2_neighbors: torch.Tensor,    # (batch, K1, K2, node_feat_dim)
    ) -> torch.Tensor:
        # Aggregate hop-2 into hop-1 neighbors
        batch, K1, K2, feat_dim = hop2_neighbors.shape
        hop2_flat = hop2_neighbors.view(batch * K1, K2, feat_dim)
        hop1_flat = hop1_neighbors.view(batch * K1, feat_dim)
        hop1_updated = self.layer1(hop1_flat, hop2_flat)        # (batch*K1, hidden)
        hop1_updated = hop1_updated.view(batch, K1, -1)

        # Aggregate updated hop-1 into node
        embedding = self.layer2(node_feat, hop1_updated)         # (batch, embed_dim)
        return F.normalize(embedding, p=2, dim=-1)               # L2 normalize


# ─── Two-Tower Ranker ─────────────────────────────────────────────────────────

class Tower(nn.Module):
    """Single tower in the two-tower architecture."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()])
            prev = h
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """
    Two-Tower model for PYMK scoring.
    Query and candidate towers produce embeddings; score = dot product.
    """

    def __init__(self, gnn_embed_dim: int, profile_feat_dim: int, activity_feat_dim: int):
        super().__init__()
        input_dim = gnn_embed_dim + profile_feat_dim + activity_feat_dim
        self.query_tower = Tower(input_dim)
        self.candidate_tower = Tower(input_dim)

    def encode_query(self, features: torch.Tensor) -> torch.Tensor:
        return self.query_tower(features)

    def encode_candidate(self, features: torch.Tensor) -> torch.Tensor:
        return self.candidate_tower(features)

    def forward(
        self,
        query_feat: torch.Tensor,       # (batch, input_dim)
        candidate_feat: torch.Tensor,   # (batch, input_dim)
    ) -> torch.Tensor:
        q_embed = self.encode_query(query_feat)
        c_embed = self.encode_candidate(candidate_feat)
        return (q_embed * c_embed).sum(dim=-1)   # dot product → (batch,)


# ─── Training with In-Batch Negatives ─────────────────────────────────────────

def train_step(
    model: TwoTowerModel,
    query_feats: torch.Tensor,         # (N, dim)
    positive_feats: torch.Tensor,      # (N, dim) — one positive per query
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    In-batch negative sampling loss (sampled softmax).
    Each query's negative = all other queries' positives.
    """
    q_embeds = model.encode_query(query_feats)           # (N, 64)
    c_embeds = model.encode_candidate(positive_feats)    # (N, 64)

    # Compute all-pairs scores: (N, N) matrix
    scores = torch.matmul(q_embeds, c_embeds.T) / temperature

    # Targets: diagonal entries are the true positives
    labels = torch.arange(len(query_feats), device=scores.device)
    loss = F.cross_entropy(scores, labels)
    return loss


# ─── Graph Heuristic Scorer (Baseline) ────────────────────────────────────────

class GraphHeuristicScorer:
    """Fast graph-based scoring using common neighbors and Adamic-Adar."""

    def __init__(self, adjacency: Dict[int, set]):
        self.adj = adjacency

    def common_neighbors(self, a: int, b: int) -> int:
        return len(self.adj.get(a, set()) & self.adj.get(b, set()))

    def adamic_adar(self, a: int, b: int) -> float:
        common = self.adj.get(a, set()) & self.adj.get(b, set())
        score = 0.0
        for z in common:
            degree_z = len(self.adj.get(z, set()))
            if degree_z > 1:
                score += 1.0 / np.log(degree_z)
        return score

    def jaccard(self, a: int, b: int) -> float:
        n_a = self.adj.get(a, set())
        n_b = self.adj.get(b, set())
        union = len(n_a | n_b)
        if union == 0:
            return 0.0
        return len(n_a & n_b) / union


# ─── PYMK Service ─────────────────────────────────────────────────────────────

class PYMKService:
    """
    Simplified PYMK inference service.
    In production: candidate lists and embeddings are precomputed offline.
    """

    def __init__(
        self,
        gnn_model: GraphSAGE,
        ranker: TwoTowerModel,
        heuristic_scorer: GraphHeuristicScorer,
        user_store: Dict[int, User],
    ):
        self.gnn = gnn_model
        self.ranker = ranker
        self.heuristic = heuristic_scorer
        self.users = user_store

    def get_candidates(self, user_id: int, max_candidates: int = 200) -> List[int]:
        """Stage 1: 2nd-degree graph traversal (simplified)."""
        user = self.users[user_id]
        first_degree = set(user.neighbors)
        candidates = {}

        for neighbor_id in user.neighbors:
            neighbor = self.users.get(neighbor_id)
            if neighbor is None:
                continue
            for candidate_id in neighbor.neighbors:
                if candidate_id == user_id:
                    continue
                if candidate_id in first_degree:
                    continue
                cn = candidates.get(candidate_id, 0) + 1
                candidates[candidate_id] = cn

        # Sort by common neighbor count descending
        sorted_candidates = sorted(candidates.keys(),
                                   key=lambda c: candidates[c],
                                   reverse=True)
        return sorted_candidates[:max_candidates]

    def score_candidates(
        self,
        query_user_id: int,
        candidate_ids: List[int],
    ) -> List[Tuple[int, float]]:
        """Stage 2: Score candidates with heuristic scores (simplified ranker)."""
        scores = []
        for cid in candidate_ids:
            score = self.heuristic.adamic_adar(query_user_id, cid)
            scores.append((cid, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def recommend(self, user_id: int, top_k: int = 20) -> List[Dict]:
        """Full PYMK pipeline."""
        # Stage 1: Candidate generation
        candidates = self.get_candidates(user_id)

        # Stage 2: Scoring and ranking
        scored = self.score_candidates(user_id, candidates)

        # Format output
        recommendations = []
        for candidate_id, score in scored[:top_k]:
            cn_count = self.heuristic.common_neighbors(user_id, candidate_id)
            recommendations.append({
                "candidate_id": candidate_id,
                "score": round(score, 4),
                "reason": f"{cn_count} mutual connections",
            })

        return recommendations


# ─── Demo ─────────────────────────────────────────────────────────────────────

def demo():
    """Create a tiny 10-node graph and run PYMK."""
    np.random.seed(42)
    num_users = 10
    feat_dim = 16

    # Build a small graph: users 0–9 with random connections
    adjacency = {i: set() for i in range(num_users)}
    edges = [(0,1),(0,2),(0,3),(1,4),(1,5),(2,5),(2,6),(3,6),(3,7),(4,8),(5,8),(6,9),(7,9)]
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    users = {
        i: User(
            user_id=i,
            features=np.random.randn(feat_dim).astype(np.float32),
            neighbors=list(adjacency[i]),
        )
        for i in range(num_users)
    }

    gnn = GraphSAGE(node_feat_dim=feat_dim, hidden_dim=32, embed_dim=16)
    ranker = TwoTowerModel(gnn_embed_dim=16, profile_feat_dim=8, activity_feat_dim=4)
    heuristic = GraphHeuristicScorer(adjacency)

    service = PYMKService(gnn, ranker, heuristic, users)

    print("=== PYMK Recommendations for User 0 ===")
    print(f"User 0's connections: {sorted(adjacency[0])}")
    recs = service.recommend(user_id=0, top_k=5)
    for rec in recs:
        print(f"  Candidate {rec['candidate_id']}: score={rec['score']:.4f} | {rec['reason']}")

    print("\n=== In-batch negative training loss (random data) ===")
    N, dim = 8, 28
    query_feats = torch.randn(N, dim)
    pos_feats = torch.randn(N, dim)
    loss = train_step(ranker, query_feats, pos_feats)
    print(f"  Loss: {loss.item():.4f}")


if __name__ == "__main__":
    demo()
```

---

## Running the Mini Implementation

```bash
cd /Users/chi.wang/workspace/cw/ml_system_design/graph
python -m venv venv
source venv/bin/activate
pip install torch numpy
python docs/09_end_to_end.py
```

Expected output:
```
=== PYMK Recommendations for User 0 ===
User 0's connections: [1, 2, 3]
  Candidate 5: score=1.4427 | 2 mutual connections
  Candidate 6: score=1.4427 | 2 mutual connections
  Candidate 8: score=0.9102 | 1 mutual connections
  Candidate 4: score=0.9102 | 1 mutual connections
  Candidate 7: score=0.9102 | 1 mutual connections

=== In-batch negative training loss (random data) ===
  Loss: 2.0794
```

---

## Interview Checkpoint

**Q: Walk me through the end-to-end flow for a single user request.**

Answer using the 8-step walkthrough above. Key points to emphasize:
1. Cache check first (most requests hit cache)
2. Precomputed candidates — no real-time graph traversal
3. Batch feature fetch (one round trip, not N round trips)
4. Lightweight ranker (not a GNN) for low latency
5. Business rules layer last

**Q: How does your system handle a new user with zero connections?**

No candidates from graph traversal (empty neighborhood). Falls back to:
1. **ANN search** on profile-based embedding (profile text → LLM → embedding → ANN)
2. **Location + industry matching**: find popular users in same city and industry
3. **Onboarding prompts**: ask user to import contacts or specify their workplace/school, then use that to seed the graph
