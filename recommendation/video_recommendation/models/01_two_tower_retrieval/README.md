# Two-Tower Retrieval Model (Collaborative Filtering)

## 1. Role in the Pipeline

The Two-Tower model is the **primary retrieval source** in Stage 1 (Candidate Generation). It retrieves ~100 candidates from a corpus of millions by learning user and item embeddings in a shared vector space, then performing approximate nearest neighbor (ANN) search.

**Why Two-Tower?**
- Decouples user and item encoding → item embeddings can be **precomputed offline**
- ANN search over precomputed item embeddings gives **sub-10ms** retrieval at scale
- Used at: **YouTube (2019 DNN retrieval), Google, Facebook, Pinterest**

```
┌──────────────┐     ┌──────────────┐
│  User Tower   │     │  Item Tower   │
│  (query enc.) │     │ (video enc.)  │
└──────┬───────┘     └──────┬───────┘
       │                     │
   user_emb (d=128)     item_emb (d=128)
       │                     │
       └──────┬──────────────┘
              │
        dot product / cosine
              │
         similarity score
```

---

## 2. Data Requirements

### Training Data

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | string | Unique user identifier |
| `video_id` | string | Unique video identifier |
| `label` | int | 1 = positive interaction, 0 = negative |
| `watch_time_sec` | float | Seconds watched (for weighted positives) |
| `timestamp` | int64 | Event time (for train/val split) |

### Positive Signals (label = 1)
- User watched video > 50% of its duration
- User liked the video
- User shared the video

### Negative Sampling Strategy

| Strategy | Description | Trade-off |
|----------|-------------|-----------|
| **Random negatives** (most common) | Sample random videos the user didn't interact with | Simple, but may include easy negatives |
| **In-batch negatives** | Other users' positives in the same batch become negatives | Free negatives, efficient GPU usage. **Used by YouTube, Google** |
| **Hard negatives** | Videos user was shown but didn't click | More informative, but risk selection bias |
| **Mixed** | Combine random + hard negatives | Best of both worlds; typical ratio is 80% random, 20% hard |

**Recommended**: In-batch negatives + a small fraction of hard negatives.

### Scale
- Training set: ~100M–1B interaction pairs
- User corpus: ~10M–100M users
- Video corpus: ~10M–100M videos

### Example Raw Data

```json
{"user_id": "u_382910", "video_id": "v_19283", "label": 1, "watch_time_sec": 187.5, "timestamp": 1700000000}
{"user_id": "u_382910", "video_id": "v_88421", "label": 0, "watch_time_sec": 3.2, "timestamp": 1700000100}
{"user_id": "u_501223", "video_id": "v_19283", "label": 1, "watch_time_sec": 210.0, "timestamp": 1700000200}
```

---

## 3. Feature Engineering

### User Tower Features

| Feature | Type | Processing | Dimension |
|---------|------|------------|-----------|
| `user_id` | categorical | Embedding lookup | 64 |
| `age_bucket` | categorical | Bucketize [0-17, 18-24, 25-34, 35-44, 45-54, 55+] | 8 |
| `country` | categorical | Embedding lookup (top 50 + "other") | 16 |
| `language` | categorical | Embedding lookup | 8 |
| `device_type` | categorical | One-hot [mobile, tablet, desktop, TV] | 4 |
| `avg_watch_time_7d` | continuous | Log-transform, normalize | 1 |
| `num_videos_watched_7d` | continuous | Log-transform, normalize | 1 |
| `watch_history_emb` | sequence | Average of last 50 watched video embeddings | 128 |
| `category_affinity` | vector | Normalized watch-time distribution over categories | 20 |
| `hour_of_day` | categorical | Embedding | 8 |
| `day_of_week` | categorical | Embedding | 4 |

### Item (Video) Tower Features

| Feature | Type | Processing | Dimension |
|---------|------|------------|-----------|
| `video_id` | categorical | Embedding lookup | 64 |
| `creator_id` | categorical | Embedding lookup | 32 |
| `category` | categorical | Embedding lookup | 16 |
| `duration_bucket` | categorical | Bucketize [<1m, 1-5m, 5-15m, 15-30m, 30m+] | 8 |
| `upload_age_days` | continuous | Log-transform, normalize | 1 |
| `title_emb` | dense | Pretrained text embedding (BERT/sentence-transformers) | 128 |
| `total_views_log` | continuous | Log(views + 1) | 1 |
| `avg_completion_rate` | continuous | Raw [0, 1] | 1 |
| `like_ratio` | continuous | likes / (likes + dislikes + 1) | 1 |

### Feature Engineering Key Decisions

**Why embed watch history as an average?**
- Scalable: O(1) lookup at serving time if pre-aggregated
- Alternative: Use attention (transformer) over history → better quality but higher serving cost
- YouTube's 2016 paper used average pooling of watch history embeddings

**Why log-transform continuous features?**
- Watch time, view count follow power-law distributions
- Log-transform reduces skew, improves gradient dynamics

---

## 4. Model Architecture

### Option A: Simple Two-Tower (Recommended Starting Point)

```
User Tower:                          Item Tower:
  concat(user features)                concat(item features)
       ↓                                    ↓
  Dense(512) + ReLU + BN               Dense(512) + ReLU + BN
       ↓                                    ↓
  Dense(256) + ReLU + BN               Dense(256) + ReLU + BN
       ↓                                    ↓
  Dense(128) → L2 normalize            Dense(128) → L2 normalize
       ↓                                    ↓
   user_emb (128-d)                    item_emb (128-d)
```

Score = dot_product(user_emb, item_emb)

### Option B: Two-Tower with Cross-Attention on History

Replace average pooling of watch history with a target-aware attention layer:
- Attend over recent watch history **conditioned on the candidate item**
- Better captures relevance-dependent user interests
- Trade-off: Cannot fully precompute user embedding (requires candidate info)

### Option C: YouTube DNN Retrieval (YouTube 2016 Paper)

- Single tower that takes both user and context features
- Predicts next video watched as a multi-class classification
- Uses softmax over entire video corpus (sampled softmax in practice)
- Simpler but doesn't separate user/item computation cleanly

### Comparison

| Aspect | Simple Two-Tower | Cross-Attention | YouTube DNN |
|--------|-----------------|-----------------|-------------|
| Serving latency | Lowest (precomputed items) | Medium (attention at serve time) | Medium |
| Quality | Good | Best | Good |
| Offline compute | Item embs precomputed | Partial precompute | Item embs precomputed |
| Complexity | Low | High | Medium |
| **Recommendation** | **Production default** | Large teams | Historical reference |

---

## 4.1 Beyond Two-Tower: Cross-Attention & Late Interaction

### The Core Limitation of Two-Tower

Two-Tower **must produce user and item embeddings independently** — the user tower never sees the candidate item, and vice versa. This is what makes ANN precomputation possible, but it's also a fundamental expressiveness bottleneck.

```
Two-Tower (independent encoding):

  User features ──→ User Tower ──→ user_emb ─┐
                                              ├── dot product → score
  Item features ──→ Item Tower ──→ item_emb ─┘

  Problem: the user embedding is the SAME regardless of which item we're scoring.
  A single 128-d vector must simultaneously encode:
    - "I like short gaming clips"
    - "I like long ML tutorials"
    - "I like cooking at night"
  All compressed into one point in space → information loss.
```

**Concrete example of what Two-Tower misses:**

```
User has multi-faceted interests: gaming (40%), ML tutorials (30%), cooking (30%)

Two-Tower: user_emb = average of all interests → a point somewhere
           between gaming, ML, and cooking clusters

  Candidate A: "Advanced PyTorch Tips" (ML tutorial)
    → dot product with averaged user_emb → moderate score

  Candidate B: "Fortnite Season 8 Highlights" (gaming)
    → dot product with averaged user_emb → moderate score

  Both get similar scores because the single user vector
  dilutes each interest. The model can't "activate" the ML part
  of the user when scoring an ML video.

Cross-Attention: When scoring Candidate A, attend over the user's
  watch history → ML tutorial history items get HIGH attention weights
  → user representation is dynamically weighted toward ML → high score

  When scoring Candidate B, the SAME history → gaming items get high
  attention weights → user representation shifts toward gaming → high score
```

### What Is Cross-Attention?

Cross-attention allows one tower to **look at the other side's features** before producing the final score. The key idea: the user representation should be **different for each candidate item**.

```
Standard Two-Tower (no cross-interaction):

  User features → [DNN] → user_emb (fixed)
  Item features → [DNN] → item_emb (fixed)
  Score = dot(user_emb, item_emb)


Cross-Attention (target-aware user representation):

  User watch history: [h₁, h₂, h₃, ..., h₅₀]    (50 video embeddings)
  Candidate item embedding: q (from item tower)

  Attention weights: αᵢ = softmax(q · hᵢ)         ← item queries the history
  Attended user repr: u_attended = Σ αᵢ · hᵢ       ← weighted by relevance to THIS item

  Final user_emb = DNN(user_features, u_attended)   ← candidate-aware
  Score = dot(user_emb, item_emb)
```

**The attention mechanism in detail:**

```
User's watch history embeddings (each 128-d):

  h₁ = "Minecraft Let's Play"        [0.8, 0.1, -0.2, ...]
  h₂ = "PyTorch Tutorial"            [0.1, 0.9, 0.3, ...]
  h₃ = "CS:GO Pro Match"             [0.7, 0.2, -0.1, ...]
  h₄ = "Neural Network Math"         [0.1, 0.8, 0.4, ...]
  h₅ = "Easy Pasta Recipe"           [-0.1, 0.1, 0.9, ...]

Candidate: "TensorFlow vs PyTorch" → item_emb = [0.1, 0.85, 0.3, ...]

Attention scores (dot product with candidate):
  α₁ = dot(item_emb, h₁) = 0.08 + 0.085 - 0.06  = 0.105  (low — gaming)
  α₂ = dot(item_emb, h₂) = 0.01 + 0.765 + 0.09  = 0.865  (HIGH — ML)
  α₃ = dot(item_emb, h₃) = 0.07 + 0.17  - 0.03  = 0.210  (low — gaming)
  α₄ = dot(item_emb, h₄) = 0.01 + 0.68  + 0.12  = 0.810  (HIGH — ML)
  α₅ = dot(item_emb, h₅) = -0.01 + 0.085 + 0.27 = 0.345  (medium — cooking)

After softmax: α = [0.05, 0.35, 0.08, 0.33, 0.19]

Attended representation: heavily weighted toward ML history items
→ user_emb shifts toward ML space → HIGH match with this ML candidate
```

### Architecture Options

#### Option 1: Target-Aware Attention (Most Common)

Item embedding queries the user's behavior sequence. Used by **Alibaba (DIN), YouTube**.

```
                User Tower                              Item Tower

  user_profile ───→ [MLP] ──┐                item features → [MLP] → item_emb
                             │                                          │
  watch_history              │                                          │
  [h₁, h₂, ..., h₅₀]       │                                          │
       │                     │         ┌────────────────────────────────┘
       │                     │         │ (item_emb used as attention query)
       ▼                     │         ▼
  ┌──────────────────┐       │    ┌─────────┐
  │  Attention Layer  │◄──────────│ q = item │
  │                   │       │    │   emb   │
  │  αᵢ = softmax(   │       │    └─────────┘
  │    q · hᵢ / √d)  │       │
  │                   │       │
  │  out = Σ αᵢ · hᵢ │       │
  └────────┬─────────┘       │
           │                  │
           ▼                  │
      [concat] ◄──────────────┘
           │
      [MLP layers]
           │
       user_emb (candidate-aware, 128-d)
           │
           └───── dot product with item_emb ──→ score
```

#### Option 2: Multi-Head Cross-Attention (Transformer-Style)

Multiple attention heads capture different interest aspects.

```
  Q = item_emb projected through W_Q                  (1, d_k) × num_heads
  K = watch_history projected through W_K              (50, d_k) × num_heads
  V = watch_history projected through W_V              (50, d_v) × num_heads

  Attention = softmax(Q × K^T / √d_k) × V             per head
  Output = concat(head_1, ..., head_8) × W_O           (1, 128)
```

#### Option 3: Late Interaction (ColBERT-style)

Keep **multiple** embeddings per user (not compressed to one vector) and compute fine-grained token-level similarity. Originated in information retrieval (ColBERT), adapted for recommendations.

```
Standard Two-Tower:
  user → 1 embedding (128-d)
  item → 1 embedding (128-d)
  Score = dot(user_emb, item_emb)                    → 1 comparison

Late Interaction:
  user → N embeddings [u₁, u₂, ..., u₅₀]            (from watch history)
  item → M embeddings [v₁, v₂, ..., v₅]              (from title tokens, etc.)

  Score = Σᵢ maxⱼ dot(uᵢ, vⱼ)                        → N×M comparisons
          (each user token finds its best-matching item token)
```

### Architecture Comparison (Extended)

| Aspect | Simple Two-Tower | Target-Aware Attention | Multi-Head Cross-Attn | Late Interaction |
|--------|-----------------|----------------------|----------------------|-----------------|
| User-item interaction | None | Item queries user history | Full cross-attention | Token-level matching |
| User emb depends on item? | No | **Yes** | **Yes** | **Yes** |
| Can precompute item embs? | Yes | Yes (item tower is independent) | Yes | Yes |
| Can precompute user embs? | **Yes** | **No** (changes per candidate) | **No** | Partially |
| ANN retrieval possible? | **Yes** | **No** (must score each candidate) | **No** | Partially (with approximations) |
| Retrieval latency | ~5ms (ANN) | ~50ms (score N candidates) | ~50ms | ~20ms |
| Quality | Good | Better | Best | Better |
| Used at | YouTube retrieval | **Alibaba DIN**, YouTube ranking | Research | ColBERT (search) |
| **Best stage** | **Retrieval** | **Ranking** (or hybrid) | **Ranking** | **Ranking / Re-ranking** |

### The Serving Dilemma: Why This Matters

The fundamental tension:

```
              Can precompute user emb?
              │
         ┌────┴─────┐
         Yes         No
         │           │
    ANN possible   Must score each
    (fast, O(1))   candidate individually
         │           │
    ┌────┴────┐     ┌────┴─────┐
    Two-Tower   Can't use       Cross-Attention
    Retrieval   at retrieval    Ranking
    (~5ms)      stage           (~50ms for 300 items)
```

**This is why cross-attention models are typically used in the ranking stage, not retrieval.**

At retrieval, you need to search millions of items — only ANN (which requires precomputed, fixed embeddings) can do this in milliseconds. At ranking, you only score ~300 candidates, so the per-candidate cost of cross-attention is affordable.

### Hybrid Approach: Two-Tower Retrieval + Cross-Attention Ranking

The production solution at companies like YouTube, TikTok, and Alibaba:

```
Stage 1 — Retrieval (Two-Tower):
  10M videos → precomputed item_embs in FAISS
  User features → user_tower → user_emb (FIXED, no cross-attention)
  ANN search → 300 candidates in ~5ms
  Quality: good enough to not miss relevant items (Recall@300 > 0.5)

Stage 2 — Ranking (Cross-Attention / MMoE):
  300 candidates × full feature vectors
  For each candidate: cross-attention over user history conditioned on THIS item
  Rich per-candidate scoring → 5 engagement predictions
  Latency: ~50ms for 300 items on GPU

The two stages complement each other:
  • Retrieval: coarse but fast (cast a wide net)
  • Ranking: precise but slow (pick the best from the net)
```

### Code: Target-Aware Attention Two-Tower

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetAwareAttention(nn.Module):
    """
    Target-aware attention: the candidate item attends over
    the user's watch history to produce a candidate-specific
    user representation.

    Based on DIN (Deep Interest Network, Alibaba 2018).
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_Q = nn.Linear(embed_dim, embed_dim)  # Query from candidate item
        self.W_K = nn.Linear(embed_dim, embed_dim)  # Key from history items
        self.W_V = nn.Linear(embed_dim, embed_dim)  # Value from history items
        self.W_O = nn.Linear(embed_dim, embed_dim)

    def forward(self, item_emb: torch.Tensor,
                history_embs: torch.Tensor,
                history_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            item_emb:      (B, D)       candidate item embedding
            history_embs:  (B, L, D)    user's last L watched video embeddings
            history_mask:  (B, L)       1 = real item, 0 = padding

        Returns:
            attended_repr: (B, D)       candidate-aware user representation
        """
        B, L, D = history_embs.shape

        # Project to Q, K, V
        Q = self.W_Q(item_emb).unsqueeze(1)   # (B, 1, D)
        K = self.W_K(history_embs)              # (B, L, D)
        V = self.W_V(history_embs)              # (B, L, D)

        # Reshape for multi-head attention
        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, d)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, d)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, d)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # scores: (B, H, 1, L)

        # Mask padding positions
        if history_mask is not None:
            mask = history_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)  # (B, H, 1, L)
        attended = torch.matmul(attn_weights, V)   # (B, H, 1, d)

        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(B, 1, D)  # (B, 1, D)
        attended = self.W_O(attended).squeeze(1)  # (B, D)

        return attended


class CrossAttentionTwoTower(nn.Module):
    """
    Two-Tower with target-aware attention on user watch history.

    User tower: user_profile_features + attention(history, candidate_item)
    Item tower: standard MLP (independent, precomputable)

    NOTE: user_emb now depends on the candidate → cannot use ANN for retrieval.
    Best used for ranking or a small candidate set.
    """

    def __init__(self, user_profile_dim: int, item_feat_dim: int,
                 history_emb_dim: int = 128, embed_dim: int = 128):
        super().__init__()

        # Item tower (independent — can be precomputed)
        self.item_tower = nn.Sequential(
            nn.Linear(item_feat_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

        # Attention layer (item queries user history)
        self.attention = TargetAwareAttention(history_emb_dim, num_heads=4)

        # User tower (takes profile features + attended history)
        self.user_tower = nn.Sequential(
            nn.Linear(user_profile_dim + embed_dim, 512),  # profile + attended
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, user_profile: torch.Tensor,
                user_history: torch.Tensor,
                history_mask: torch.Tensor,
                item_features: torch.Tensor) -> dict:
        """
        Args:
            user_profile: (B, user_profile_dim) non-sequence user features
            user_history: (B, L, 128) embeddings of last L watched videos
            history_mask: (B, L) padding mask
            item_features: (B, item_feat_dim) candidate item features
        """
        # Item tower (independent)
        item_emb = self.item_tower(item_features)           # (B, D)
        item_emb = F.normalize(item_emb, p=2, dim=1)

        # Attention: item queries user's history
        attended_history = self.attention(item_emb, user_history, history_mask)

        # User tower: profile + candidate-aware history summary
        user_input = torch.cat([user_profile, attended_history], dim=1)
        user_emb = self.user_tower(user_input)              # (B, D)
        user_emb = F.normalize(user_emb, p=2, dim=1)

        # Score
        score = (user_emb * item_emb).sum(dim=1)            # (B,)

        return {
            "score": score,
            "user_emb": user_emb,
            "item_emb": item_emb,
        }
```

### Practical Example: How Attention Changes the User Embedding

```python
# Same user, two different candidate items → two different user embeddings

user_history = [
    "Minecraft Stream",        # gaming
    "PyTorch Deep Dive",       # ML
    "League of Legends Tips",  # gaming
    "Transformer Architecture",# ML
    "Pasta Carbonara Recipe",  # cooking
]

# Candidate A: "Advanced CUDA Programming" (ML/systems)
attended_A = attention(candidate_A_emb, user_history)
# Attention weights: [0.05, 0.38, 0.04, 0.42, 0.11]
#                     game  ML    game  ML    cook
# → user_emb_A is pulled toward ML/systems space
# → HIGH score with "Advanced CUDA Programming"

# Candidate B: "Fortnite Chapter 5 Review" (gaming)
attended_B = attention(candidate_B_emb, user_history)
# Attention weights: [0.35, 0.05, 0.40, 0.04, 0.16]
#                     game  ML    game  ML    cook
# → user_emb_B is pulled toward gaming space
# → HIGH score with "Fortnite Chapter 5 Review"

# Standard Two-Tower: BOTH candidates would see the SAME user_emb
# (average of all interests) → both get a mediocre moderate score
```

### When To Use What (Decision Framework)

```
                   Need to search millions of items?
                   │
              ┌────┴─────┐
              Yes         No (already have < 1000 candidates)
              │           │
        Must use ANN      Can afford per-candidate scoring
              │           │
        Two-Tower         Cross-Attention or full interaction
        (retrieval)       (ranking stage)
              │           │
              │      ┌────┴──────────┐
              │      │               │
              │    Few hundred      Tens of candidates
              │    candidates       (re-ranking)
              │      │               │
              │    Target-Aware     Late Interaction
              │    Attention        or full Transformer
              │    (DIN-style)
              │
         Can we get some cross-attention benefit
         while keeping ANN possible?
              │
         ┌────┴────┐
         │         │
    Multi-Interest  Approximate methods
    (MIND/Alibaba)  (precompute multiple
                     user embeddings,
                     query each via ANN)
```

### Key Papers & Industry References

| Approach | Paper / System | Year | Key Idea |
|----------|---------------|------|----------|
| Two-Tower (DSSM) | Huang et al., Microsoft | 2013 | Dual encoder with dot product |
| YouTube DNN | Covington et al., Google | 2016 | Deep retrieval with sampled softmax |
| DIN (Deep Interest Network) | Zhou et al., Alibaba | 2018 | Target-aware attention over history |
| DIEN (Deep Interest Evolution) | Zhou et al., Alibaba | 2019 | GRU + attention to model interest evolution |
| MIND (Multi-Interest Network) | Li et al., Alibaba | 2019 | Capsule network → multiple user embeddings for ANN |
| SDM (Sequential Deep Matching) | Lv et al., Alibaba | 2019 | Combine long/short-term history with attention |
| SASRec (Self-Attentive) | Kang & McAuley | 2018 | Transformer over user sequence |
| ColBERT | Khattab & Zaharia | 2020 | Late interaction for retrieval (IR, adaptable to recs) |

### Interview Talking Points

1. **"Why not use cross-attention in retrieval?"**
   > Because the user embedding becomes candidate-dependent, so you can't precompute a single user vector for ANN search over millions of items. Each candidate would require a separate forward pass through the user tower — infeasible at retrieval scale but fine for ranking ~300 candidates.

2. **"How to get the benefit of cross-attention in retrieval?"**
   > Multi-Interest Networks (MIND): produce K user embeddings (e.g., K=5 interest clusters), run K ANN queries, merge results. This captures multi-faceted interests without full cross-attention. Alibaba uses this in production.

3. **"What's the biggest quality gap between Two-Tower and cross-attention?"**
   > Multi-interest users. A single embedding must compress gaming + cooking + ML interests into one point. Cross-attention dynamically activates the relevant interest per candidate. In practice, adding target-aware attention to the ranking stage gives 2-5% improvement in engagement metrics.

4. **"DIN vs. Transformer-based attention?"**
   > DIN uses a simple MLP-based attention (candidate queries history). SASRec and Transformers use self-attention over the sequence first (model interest evolution), then optionally cross-attend with the candidate. Transformers are more powerful but more expensive. For ranking with ~300 candidates, the compute difference is manageable.

---

## 5. High-Level Training Code (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Tower(nn.Module):
    """Shared tower architecture for both user and item encoders."""

    def __init__(self, input_dim: int, embedding_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.net(x)
        return F.normalize(emb, p=2, dim=1)  # L2 normalize


class TwoTowerModel(nn.Module):
    """Two-Tower retrieval model with in-batch negatives."""

    def __init__(self, user_feat_dim: int, item_feat_dim: int,
                 embedding_dim: int = 128, temperature: float = 0.05):
        super().__init__()
        self.user_tower = Tower(user_feat_dim, embedding_dim)
        self.item_tower = Tower(item_feat_dim, embedding_dim)
        self.temperature = temperature

    def forward(self, user_features: torch.Tensor,
                item_features: torch.Tensor) -> dict:
        """
        Args:
            user_features: (batch_size, user_feat_dim)
            item_features: (batch_size, item_feat_dim)

        Returns:
            dict with user_emb, item_emb, logits
        """
        user_emb = self.user_tower(user_features)   # (B, D)
        item_emb = self.item_tower(item_features)    # (B, D)

        # In-batch negative logits: each user scored against all items in batch
        # logits[i][j] = similarity(user_i, item_j)
        logits = torch.matmul(user_emb, item_emb.T) / self.temperature  # (B, B)

        return {
            "user_emb": user_emb,
            "item_emb": item_emb,
            "logits": logits,
        }


def train_two_tower(model, train_loader, optimizer, device, epochs=5):
    """Training loop with in-batch negative softmax loss."""
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            user_feat = batch["user_features"].to(device)
            item_feat = batch["item_features"].to(device)

            output = model(user_feat, item_feat)

            # In-batch negatives: positive pair is on the diagonal
            # labels[i] = i (user_i should match item_i)
            labels = torch.arange(user_feat.size(0), device=device)
            loss = F.cross_entropy(output["logits"], labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: avg_loss={total_loss/len(train_loader):.4f}")


# ---------- Inference: Building the ANN Index ----------

def build_item_index(model, item_dataloader, device):
    """Precompute all item embeddings and build FAISS index."""
    import faiss

    model.eval()
    all_embeddings = []
    all_video_ids = []

    with torch.no_grad():
        for batch in item_dataloader:
            item_feat = batch["item_features"].to(device)
            item_emb = model.item_tower(item_feat)
            all_embeddings.append(item_emb.cpu().numpy())
            all_video_ids.extend(batch["video_id"])

    import numpy as np
    embeddings = np.vstack(all_embeddings).astype("float32")

    # Build FAISS index (IVF for large-scale, flat for small)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine after L2-norm)
    index.add(embeddings)

    return index, all_video_ids


def retrieve_candidates(model, user_features, index, video_ids,
                        device, top_k=100):
    """Retrieve top-K candidates for a user via ANN search."""
    model.eval()
    with torch.no_grad():
        user_emb = model.user_tower(user_features.to(device))
        user_emb_np = user_emb.cpu().numpy().astype("float32")

    scores, indices = index.search(user_emb_np, top_k)

    candidates = [
        {"video_id": video_ids[idx], "score": float(score)}
        for idx, score in zip(indices[0], scores[0])
    ]
    return candidates
```

---

## 6. Model Input / Output Examples

### Training Input Example (one batch row)

```
User Features (concatenated dense vector, dim=262):
┌─────────────────────────────────────────────────────────────────┐
│ user_id_emb(64) | age_bucket_emb(8) | country_emb(16) |        │
│ language_emb(8) | device_onehot(4) | avg_watch_7d(1) |          │
│ num_videos_7d(1) | watch_history_emb(128) | cat_affinity(20) |  │
│ hour_emb(8) | dow_emb(4)                                        │
└─────────────────────────────────────────────────────────────────┘

Item Features (concatenated dense vector, dim=252):
┌─────────────────────────────────────────────────────────────────┐
│ video_id_emb(64) | creator_id_emb(32) | category_emb(16) |      │
│ duration_bucket_emb(8) | upload_age(1) | title_emb(128) |       │
│ views_log(1) | completion_rate(1) | like_ratio(1)               │
└─────────────────────────────────────────────────────────────────┘
```

### Inference Input/Output

**Input**: User features for user `u_382910`
```python
user_features = {
    "user_id": "u_382910",
    "age_bucket": "25-34",
    "country": "US",
    "language": "en",
    "device": "mobile",
    "avg_watch_time_7d": 1250.0,    # seconds
    "num_videos_watched_7d": 85,
    "recent_watch_ids": ["v_100", "v_205", "v_317", ...],  # last 50
    "category_affinity": {"music": 0.3, "gaming": 0.25, "tech": 0.2, ...},
    "hour": 20,
    "day_of_week": 5,
}
```

**Output**: Top-100 candidate videos with scores
```python
candidates = [
    {"video_id": "v_48291", "score": 0.923},
    {"video_id": "v_10382", "score": 0.891},
    {"video_id": "v_77234", "score": 0.887},
    # ... 97 more
    {"video_id": "v_55102", "score": 0.412},
]
```

---

## 7. Evaluation Methods

### Offline Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Recall@K** (K=100, 200) | Fraction of user's actual future watches that appear in top-K retrieved candidates | > 0.30 at K=100 |
| **Hit Rate@K** | Fraction of users for whom at least one relevant item is in top-K | > 0.70 at K=100 |
| **MRR (Mean Reciprocal Rank)** | Average of 1/rank of first relevant item | > 0.15 |
| **NDCG@K** | Position-aware relevance scoring | Directional improvement |

**Why Recall@K is the primary metric for retrieval?**
- Retrieval's job is to **not miss relevant items** (the ranker will sort them later)
- Precision matters less at this stage since we pass ~300 candidates to ranking

### Online Metrics (A/B Test)

| Metric | Description |
|--------|-------------|
| **Ranker input coverage** | % of user's future engagements that were in the candidate set |
| **Downstream engagement** | Total watch time, likes (measured end-to-end) |
| **Retrieval latency p99** | Must stay < 20ms |

### Evaluation Protocol

```
Temporal split (NOT random):
  Train:  interactions from day 1 to day T
  Val:    interactions from day T+1 to T+3
  Test:   interactions from day T+4 to T+7
```

**Why temporal split?** Random splitting leaks future information. In production, the model always predicts the future based on past data.

### Offline Evaluation Code Sketch

```python
def evaluate_recall_at_k(model, user_loader, index, video_ids,
                          ground_truth, k=100):
    """Compute Recall@K for retrieval model."""
    recalls = []

    for batch in user_loader:
        user_feat = batch["user_features"]
        user_ids = batch["user_id"]

        candidates = retrieve_candidates(model, user_feat, index,
                                          video_ids, device, top_k=k)

        for uid, cands in zip(user_ids, candidates):
            retrieved_set = set(c["video_id"] for c in cands)
            relevant_set = set(ground_truth[uid])  # videos user actually watched

            if len(relevant_set) > 0:
                recall = len(retrieved_set & relevant_set) / len(relevant_set)
                recalls.append(recall)

    return sum(recalls) / len(recalls)
```

---

## 8. Interview Talking Points

1. **Why Two-Tower over matrix factorization?**
   - Two-Tower can incorporate rich side features (not just ID embeddings)
   - Non-linear interactions via deep layers
   - Same serving pattern (precomputed item embs + ANN)

2. **In-batch negatives trade-off**
   - Pro: No extra negative sampling needed, scales with batch size
   - Con: Popular items appear as negatives more often → popularity bias
   - Fix: Apply log-frequency correction to the loss

3. **Cold-start handling**
   - New users: Fall back to popularity/trending retrieval source
   - New videos: Item tower can still compute embedding from metadata + title (no ID embedding needed initially)

4. **Serving architecture**
   - Item embeddings refreshed every few hours via batch pipeline
   - User embeddings computed in real-time at request time
   - FAISS/ScaNN index updated periodically (full rebuild or incremental)

5. **Scaling challenge: vocabulary size**
   - 100M video IDs × 64-dim embedding = ~25 GB just for ID embeddings
   - Solutions: Feature hashing, hash embeddings, or use content features only
