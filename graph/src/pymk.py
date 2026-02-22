"""
People You May Know (PYMK) — Mini Implementation
=================================================
Demonstrates the full two-stage pipeline:
  Stage 1: Candidate Generation (graph traversal + Adamic-Adar scoring)
  Stage 2: Ranking (GraphSAGE embeddings + Two-Tower MLP)

Run:
    python src/pymk.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, field


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class User:
    user_id: int
    profile_features: np.ndarray    # industry, title, location, skills (encoded)
    neighbors: Set[int] = field(default_factory=set)


# ─── Graph Heuristic Scorer ───────────────────────────────────────────────────

class GraphHeuristicScorer:
    """Structural link prediction scores (no ML required)."""

    def __init__(self, adjacency: Dict[int, Set[int]]):
        self.adj = adjacency

    def common_neighbors(self, a: int, b: int) -> int:
        return len(self.adj.get(a, set()) & self.adj.get(b, set()))

    def adamic_adar(self, a: int, b: int) -> float:
        """Adamic-Adar: shared rare mutual friends count more."""
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
        union_size = len(n_a | n_b)
        return len(n_a & n_b) / union_size if union_size > 0 else 0.0


# ─── GraphSAGE Encoder ────────────────────────────────────────────────────────

class GraphSAGELayer(nn.Module):
    """One GraphSAGE aggregation layer (mean aggregator)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim * 2, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(
        self,
        self_feat: torch.Tensor,          # (B, in_dim)
        neighbor_feat: torch.Tensor,      # (B, K, in_dim)
    ) -> torch.Tensor:
        agg = neighbor_feat.mean(dim=1)                            # (B, in_dim)
        combined = torch.cat([self_feat, agg], dim=-1)             # (B, 2*in_dim)
        return F.relu(self.bn(self.W(combined)))                   # (B, out_dim)


class GraphSAGE(nn.Module):
    """
    2-layer inductive GNN encoder.
    Takes node features + sampled neighbor features as input.
    Output: L2-normalized embedding per node.
    """

    def __init__(self, node_feat_dim: int, hidden: int = 128, embed_dim: int = 64):
        super().__init__()
        self.layer1 = GraphSAGELayer(node_feat_dim, hidden)
        self.layer2 = GraphSAGELayer(hidden, embed_dim)
        self.embed_dim = embed_dim

    def forward(
        self,
        node_feat: torch.Tensor,          # (B, node_feat_dim)
        hop1_neighbors: torch.Tensor,     # (B, K1, node_feat_dim)
        hop2_neighbors: torch.Tensor,     # (B, K1, K2, node_feat_dim)
    ) -> torch.Tensor:
        B, K1, K2, fd = hop2_neighbors.shape

        # Layer 1a: update each hop-1 node using its hop-2 neighborhood
        h2_flat = hop2_neighbors.view(B * K1, K2, fd)
        h1_flat = hop1_neighbors.view(B * K1, fd)
        h1_updated = self.layer1(h1_flat, h2_flat).view(B, K1, -1)  # (B, K1, hidden)

        # Layer 1b: also update the seed node using its hop-1 neighborhood
        seed_h1 = self.layer1(node_feat, hop1_neighbors)             # (B, hidden)

        # Layer 2: update seed using layer-1-updated hop-1 representations
        embedding = self.layer2(seed_h1, h1_updated)                 # (B, embed_dim)
        return F.normalize(embedding, p=2, dim=-1)


# ─── Two-Tower Ranker ─────────────────────────────────────────────────────────

class Tower(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64]):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


class TwoTowerRanker(nn.Module):
    """
    Produces embeddings for query and candidate users.
    Score = dot product of their normalized embeddings.
    """

    def __init__(self, total_input_dim: int):
        super().__init__()
        self.query_tower = Tower(total_input_dim)
        self.cand_tower = Tower(total_input_dim)

    def encode_query(self, x: torch.Tensor) -> torch.Tensor:
        return self.query_tower(x)

    def encode_candidate(self, x: torch.Tensor) -> torch.Tensor:
        return self.cand_tower(x)

    def score(
        self, query_feat: torch.Tensor, cand_feat: torch.Tensor
    ) -> torch.Tensor:
        q = self.encode_query(query_feat)
        c = self.encode_candidate(cand_feat)
        return (q * c).sum(dim=-1)   # (B,)


def in_batch_negative_loss(
    model: TwoTowerRanker,
    query_feats: torch.Tensor,
    pos_feats: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Sampled softmax loss using in-batch negatives.
    Each query's negatives = all other queries' positives (within the batch).
    """
    q_emb = model.encode_query(query_feats)          # (N, d)
    c_emb = model.encode_candidate(pos_feats)        # (N, d)
    logits = torch.matmul(q_emb, c_emb.T) / temperature   # (N, N)
    labels = torch.arange(len(query_feats))
    return F.cross_entropy(logits, labels)


# ─── Feature Builder ──────────────────────────────────────────────────────────

class FeatureBuilder:
    """
    Builds the input feature vector for the ranker by combining:
    - GNN embedding of query user
    - GNN embedding of candidate user
    - Hand-engineered graph and profile features
    """

    def __init__(self, gnn: GraphSAGE, scorer: GraphHeuristicScorer, users: Dict[int, User]):
        self.gnn = gnn
        self.scorer = scorer
        self.users = users
        # Precompute node embeddings (in production: done offline)
        self._embeddings: Dict[int, torch.Tensor] = {}

    def precompute_embeddings(self):
        """Offline step: compute GNN embeddings for all users."""
        self.gnn.eval()
        with torch.no_grad():
            for uid, user in self.users.items():
                self._embeddings[uid] = self._compute_embedding(uid)

    def _sample_neighbors(self, uid: int, k: int = 5) -> List[int]:
        neighbors = list(self.users[uid].neighbors)
        if len(neighbors) <= k:
            return neighbors
        return list(np.random.choice(neighbors, size=k, replace=False))

    def _compute_embedding(self, uid: int) -> torch.Tensor:
        user = self.users[uid]
        feat_dim = len(user.profile_features)

        node_feat = torch.tensor(user.profile_features, dtype=torch.float32).unsqueeze(0)

        hop1_ids = self._sample_neighbors(uid, k=5)
        if not hop1_ids:
            # Cold start: no neighbors → use own features as proxy
            hop1_feat = node_feat.unsqueeze(1).repeat(1, 5, 1)
            hop2_feat = hop1_feat.unsqueeze(2).repeat(1, 1, 5, 1)
        else:
            hop1_feats = []
            hop2_feats_all = []
            for n1 in hop1_ids:
                n1_feat = torch.tensor(
                    self.users[n1].profile_features, dtype=torch.float32
                )
                hop1_feats.append(n1_feat)
                hop2_ids = self._sample_neighbors(n1, k=5)
                if not hop2_ids:
                    hop2_feat_row = n1_feat.unsqueeze(0).repeat(5, 1)
                else:
                    hop2_feat_row = torch.stack([
                        torch.tensor(self.users[n2].profile_features, dtype=torch.float32)
                        for n2 in hop2_ids[:5]
                    ])
                    if len(hop2_feat_row) < 5:
                        pad = hop2_feat_row[-1:].repeat(5 - len(hop2_feat_row), 1)
                        hop2_feat_row = torch.cat([hop2_feat_row, pad])
                hop2_feats_all.append(hop2_feat_row)

            # Pad hop1 to fixed K1=5
            while len(hop1_feats) < 5:
                hop1_feats.append(hop1_feats[-1])
                hop2_feats_all.append(hop2_feats_all[-1])

            hop1_feat = torch.stack(hop1_feats).unsqueeze(0)           # (1, 5, d)
            hop2_feat = torch.stack(hop2_feats_all).unsqueeze(0)       # (1, 5, 5, d)

        return self.gnn(node_feat, hop1_feat, hop2_feat).squeeze(0)    # (embed_dim,)

    def get_embedding(self, uid: int) -> torch.Tensor:
        if uid not in self._embeddings:
            self._embeddings[uid] = self._compute_embedding(uid)
        return self._embeddings[uid]

    def build_pair_features(self, query_id: int, cand_id: int) -> torch.Tensor:
        """
        Concatenate GNN embeddings + graph + profile features for a (query, candidate) pair.
        """
        q_emb = self.get_embedding(query_id)    # (embed_dim,)
        c_emb = self.get_embedding(cand_id)     # (embed_dim,)

        # Hand-engineered features
        cn = float(self.scorer.common_neighbors(query_id, cand_id))
        aa = self.scorer.adamic_adar(query_id, cand_id)
        jc = self.scorer.jaccard(query_id, cand_id)

        # Profile overlap (dot product of normalized profile features)
        q_prof = torch.tensor(self.users[query_id].profile_features, dtype=torch.float32)
        c_prof = torch.tensor(self.users[cand_id].profile_features, dtype=torch.float32)
        profile_sim = F.cosine_similarity(q_prof.unsqueeze(0), c_prof.unsqueeze(0)).item()

        handcrafted = torch.tensor([cn, aa, jc, profile_sim], dtype=torch.float32)
        return torch.cat([q_emb, c_emb, handcrafted])   # (embed_dim*2 + 4,)


# ─── PYMK Service ─────────────────────────────────────────────────────────────

class PYMKService:
    """Orchestrates candidate generation → feature building → ranking."""

    def __init__(
        self,
        ranker: TwoTowerRanker,
        feature_builder: FeatureBuilder,
        scorer: GraphHeuristicScorer,
        users: Dict[int, User],
    ):
        self.ranker = ranker
        self.features = feature_builder
        self.scorer = scorer
        self.users = users

    def _get_candidates(self, user_id: int) -> List[int]:
        """Stage 1: 2nd-degree graph traversal, filtered."""
        first_degree = self.users[user_id].neighbors
        candidate_counts: Dict[int, int] = {}

        for n in first_degree:
            for nn in self.users.get(n, User(n, np.zeros(1))).neighbors:
                if nn == user_id or nn in first_degree:
                    continue
                candidate_counts[nn] = candidate_counts.get(nn, 0) + 1

        # Sort by common neighbor count (rough pre-filter)
        return sorted(candidate_counts, key=lambda c: candidate_counts[c], reverse=True)

    def recommend(self, user_id: int, top_k: int = 5) -> List[Dict]:
        """Full PYMK pipeline."""
        candidates = self._get_candidates(user_id)
        if not candidates:
            return []

        # Stage 2: Score all candidates with ranker
        self.ranker.eval()
        scores = []
        with torch.no_grad():
            for cid in candidates:
                feat = self.features.build_pair_features(user_id, cid).unsqueeze(0)
                score = self.ranker.score(feat, feat).item()  # simplified: same features for both towers
                scores.append((cid, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        # Format output
        output = []
        for cid, score in scores[:top_k]:
            cn = self.scorer.common_neighbors(user_id, cid)
            aa = round(self.scorer.adamic_adar(user_id, cid), 4)
            output.append({
                "candidate_id": cid,
                "rank_score": round(score, 4),
                "common_neighbors": cn,
                "adamic_adar": aa,
                "reason": f"{cn} mutual connection(s), Adamic-Adar={aa}",
            })
        return output


# ─── Demo ─────────────────────────────────────────────────────────────────────

def build_demo_graph() -> Tuple[Dict[int, Set[int]], Dict[int, User]]:
    """
    Small 10-node graph for demonstration:

    0 — 1 — 4 — 8
    |   |       |
    2 — 5 ——————┘
    |   |
    3 — 6 — 9
    |       |
    └── 7 ──┘
    """
    edges = [
        (0,1),(0,2),(0,3),
        (1,4),(1,5),
        (2,5),(2,6),
        (3,6),(3,7),
        (4,8),
        (5,8),
        (6,9),(7,9),
    ]
    adj: Dict[int, Set[int]] = {i: set() for i in range(10)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    np.random.seed(42)
    feat_dim = 8
    users = {
        i: User(
            user_id=i,
            profile_features=np.random.randn(feat_dim).astype(np.float32),
            neighbors=adj[i],
        )
        for i in range(10)
    }
    return adj, users


def main():
    print("=" * 60)
    print("  People You May Know — Mini System Demo")
    print("=" * 60)

    # Build graph
    adj, users = build_demo_graph()
    print(f"\nGraph: {len(users)} users, {sum(len(v) for v in adj.values())//2} edges")
    for uid in range(10):
        print(f"  User {uid}: connections = {sorted(adj[uid])}")

    # Initialize models
    feat_dim = 8
    embed_dim = 16
    gnn = GraphSAGE(node_feat_dim=feat_dim, hidden=32, embed_dim=embed_dim)
    scorer = GraphHeuristicScorer(adj)
    feature_builder = FeatureBuilder(gnn, scorer, users)

    # Precompute embeddings (offline step)
    print("\nPrecomputing GNN embeddings (offline)...")
    feature_builder.precompute_embeddings()
    print(f"  Embedding shape per user: {feature_builder.get_embedding(0).shape}")

    # Initialize ranker
    ranker_input_dim = embed_dim * 2 + 4   # two embeddings + 4 hand-crafted features
    ranker = TwoTowerRanker(total_input_dim=ranker_input_dim)

    # Run PYMK service
    service = PYMKService(ranker, feature_builder, scorer, users)

    for query_user in [0, 1, 2]:
        print(f"\n--- PYMK for User {query_user} ---")
        print(f"  Current connections: {sorted(adj[query_user])}")
        recs = service.recommend(query_user, top_k=5)
        if not recs:
            print("  No recommendations (no 2nd-degree connections)")
        for r in recs:
            print(f"  Recommend User {r['candidate_id']}: {r['reason']}")

    # Show graph heuristics comparison
    print("\n--- Graph Heuristic Scores: User 0 vs all others ---")
    print(f"  {'Pair':<12} {'CommonNeighbors':>17} {'AdamicAdar':>12} {'Jaccard':>9}")
    print(f"  {'-'*54}")
    for cid in range(1, 10):
        if cid in adj[0]:
            continue
        cn = scorer.common_neighbors(0, cid)
        aa = scorer.adamic_adar(0, cid)
        jc = scorer.jaccard(0, cid)
        print(f"  (0, {cid}){' ':>8} {cn:>17} {aa:>12.4f} {jc:>9.4f}")

    # Training demo (in-batch negatives)
    print("\n--- Training Demo: In-Batch Negative Loss ---")
    ranker.train()
    optimizer = torch.optim.Adam(ranker.parameters(), lr=1e-3)
    for step in range(5):
        batch_size = 8
        q_feats = torch.randn(batch_size, ranker_input_dim)
        pos_feats = torch.randn(batch_size, ranker_input_dim)
        loss = in_batch_negative_loss(ranker, q_feats, pos_feats)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"  Step {step+1}: loss = {loss.item():.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
