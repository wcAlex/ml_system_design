# Two-Tower Model Training Deep Dive

How do two separate neural networks learn to map users and videos into the **same** embedding space, so that a user embedding lands near the videos they'd enjoy?

This doc covers the complete training pipeline: from raw interaction logs to a deployed model serving ANN retrieval.

---

## 1. The Core Question: How Do Two Models Learn a Shared Space?

The short answer: **they are trained jointly as one model**, not separately.

```
                    ONE model, trained end-to-end
    ┌────────────────────────────────────────────────┐
    │                                                │
    │   User features ──→ [User Tower] ──→ user_emb  │
    │                                         │      │
    │                                    dot product  │──→ loss ──→ backprop updates BOTH towers
    │                                         │      │
    │   Item features ──→ [Item Tower] ──→ item_emb  │
    │                                                │
    └────────────────────────────────────────────────┘

Training signal: "user_A watched video_X" (positive pair)
Loss pushes:  dot(user_A_emb, video_X_emb) → HIGH
              dot(user_A_emb, video_Y_emb) → LOW   (Y = random other video)

Gradients flow through BOTH towers simultaneously.
After millions of such updates, the two towers learn to produce embeddings
in a shared space where proximity = relevance.
```

**Key insight**: The dot product `dot(user_emb, item_emb)` is differentiable with respect to both towers' parameters. When backprop computes `∂loss/∂user_tower_params` and `∂loss/∂item_tower_params`, both sets of parameters get updated to make the dot product match the training signal.

This is the same principle as word2vec (word and context embeddings share the same space), CLIP (image and text encoders share the same space), and sentence-transformers (two sentence encoders share the same space).

---

## 2. Training Data: From Raw Logs to Training Records

### 2.1 Raw Interaction Logs

YouTube/TikTok collect billions of events daily:

```
Raw event stream (Kafka / event bus):
─────────────────────────────────────
{"event": "impression", "user_id": "u_42", "video_id": "v_100", "timestamp": 1700000000, "position": 3}
{"event": "click",      "user_id": "u_42", "video_id": "v_100", "timestamp": 1700000002}
{"event": "watch",      "user_id": "u_42", "video_id": "v_100", "timestamp": 1700000180, "watch_sec": 178, "video_duration": 240}
{"event": "like",       "user_id": "u_42", "video_id": "v_100", "timestamp": 1700000181}
{"event": "impression", "user_id": "u_42", "video_id": "v_201", "timestamp": 1700000000, "position": 5}
{"event": "impression", "user_id": "u_42", "video_id": "v_302", "timestamp": 1700000000, "position": 7}
```

### 2.2 Constructing Positive Pairs

A positive pair means "this user is interested in this video." Defining "interested" is a design choice:

| Definition | Signal | Trade-off |
|-----------|--------|-----------|
| **Watched > 50% duration** | Strong engagement | Misses short-form content |
| **Watched > 30 seconds** | Good for varied content | May include accidental plays |
| **Liked or shared** | Strongest signal | Very sparse (< 5% of watches) |
| **Clicked** | High volume | Noisy — clickbait gets positives |
| **Weighted by watch time** | Continuous signal | Requires regression loss, not classification |

**YouTube's approach (2016 paper)**: Use *completed watches* as positives. They also weight by watch time in the loss, so a 10-minute watch counts more than a 30-second watch.

**Recommended**: Watched > 50% of duration OR liked/shared. This balances volume with quality.

```python
# Constructing positive pairs from raw logs
def build_positive_pairs(events_df):
    """
    Join impressions with watch events to create training pairs.

    Output: (user_id, video_id, label=1, watch_time, timestamp)
    """
    # Filter to meaningful watches
    watches = events_df[
        (events_df["event"] == "watch") &
        (events_df["watch_sec"] / events_df["video_duration"] > 0.5)  # > 50% completion
    ]

    # Or explicit positive signals
    likes = events_df[events_df["event"].isin(["like", "share"])]

    positives = pd.concat([
        watches[["user_id", "video_id", "watch_sec", "timestamp"]],
        likes[["user_id", "video_id", "timestamp"]].assign(watch_sec=0),
    ]).drop_duplicates(subset=["user_id", "video_id"])

    positives["label"] = 1
    return positives
```

### 2.3 Negative Sampling: The Most Important Training Decision

The model needs to learn "user_A does NOT want video_Y." But you only have positive signals (watches). You must construct negatives.

#### Option 1: Random Negatives

Sample random videos from the corpus as negatives for each user.

```
User u_42 watched: [v_100, v_205, v_317]

Random negatives for u_42: [v_88421, v_55012, v_91003, v_22847]
(randomly sampled videos u_42 never interacted with)
```

```python
def sample_random_negatives(user_id, positive_video_ids, all_video_ids, num_neg=4):
    """Sample random videos the user didn't interact with."""
    neg_pool = list(set(all_video_ids) - set(positive_video_ids))
    return random.sample(neg_pool, min(num_neg, len(neg_pool)))
```

**Pros**: Simple, covers the full item space.
**Cons**: Most negatives are "easy" — obviously irrelevant videos. The model doesn't learn from them efficiently.

#### Option 2: In-Batch Negatives (Industry Standard)

Within a training batch, each user's positive video becomes a negative for all other users in the batch. No extra sampling needed.

```
Batch of 4 (user, positive_video) pairs:
  (u_42,  v_100)   ← u_42 watched v_100
  (u_78,  v_205)   ← u_78 watched v_205
  (u_15,  v_317)   ← u_15 watched v_317
  (u_91,  v_442)   ← u_91 watched v_442

Similarity matrix (4×4):
              v_100   v_205   v_317   v_442
  u_42      [ 0.92,  0.31,  0.15,  0.44 ]  ← want [HIGH, low, low, low]
  u_78      [ 0.28,  0.88,  0.41,  0.22 ]  ← want [low, HIGH, low, low]
  u_15      [ 0.10,  0.35,  0.85,  0.19 ]  ← want [low, low, HIGH, low]
  u_91      [ 0.45,  0.20,  0.12,  0.91 ]  ← want [low, low, low, HIGH]

Labels = [0, 1, 2, 3]  (diagonal = positive)
Loss = cross_entropy(similarity_matrix / temperature, labels)
```

This is a **softmax over the batch**: for user_i, the probability of picking item_j should be highest when j=i.

```python
def in_batch_negative_loss(user_embs, item_embs, temperature=0.05):
    """
    In-batch negatives: each user's positive item is a negative for all other users.

    Args:
        user_embs: (B, D) L2-normalized user embeddings
        item_embs: (B, D) L2-normalized item embeddings (matched pairs)
        temperature: scaling factor (lower = sharper distribution)

    Returns:
        loss: scalar
    """
    # Compute all-pairs similarity: (B, B)
    logits = torch.matmul(user_embs, item_embs.T) / temperature

    # Labels: user_i should match item_i (diagonal)
    labels = torch.arange(logits.size(0), device=logits.device)

    # Cross-entropy: treat as B-way classification
    loss = F.cross_entropy(logits, labels)

    return loss
```

**Pros**:
- Free negatives — batch size of 4096 gives you 4095 negatives per sample
- GPU-efficient — one matrix multiply computes all scores
- Used by **YouTube, Google, Facebook, Pinterest**

**Cons**:
- Popular items appear as negatives disproportionately → popularity bias
- Batch must be large enough for enough negatives (typically 2048–8192)

**Fixing the popularity bias**:

```python
def corrected_in_batch_loss(user_embs, item_embs, item_frequencies, temperature=0.05):
    """
    Log-frequency correction for in-batch negatives.
    Popular items are over-represented as negatives; correct by subtracting log(frequency).

    From: YouTube "Sampling-Bias-Corrected Neural Modeling" (2019)
    """
    logits = torch.matmul(user_embs, item_embs.T) / temperature

    # Subtract log(frequency) to down-weight popular items as negatives
    # item_frequencies: (B,) — how often each item appears in the training data
    correction = torch.log(item_frequencies + 1e-10).unsqueeze(0)  # (1, B)
    logits = logits - correction

    labels = torch.arange(logits.size(0), device=logits.device)
    loss = F.cross_entropy(logits, labels)

    return loss
```

#### Option 3: Hard Negatives

Videos the user was *shown but didn't click*. Much more informative than random.

```
Impressions for u_42:
  v_100 → clicked and watched (POSITIVE)
  v_201 → shown but not clicked (HARD NEGATIVE)
  v_302 → shown but not clicked (HARD NEGATIVE)
```

**Pros**: Forces the model to make fine-grained distinctions.
**Cons**: Risk of *selection bias* — the user only saw items the previous model chose. Also, impression-but-no-click doesn't mean irrelevant (user might not have noticed).

#### Option 4: Mixed Negatives (Best Practice)

Combine all three. Industry best practice at YouTube/Google:

```
For each positive pair (user_i, video_i):
  - In-batch negatives: other items in the batch (free, ~4095 per sample)
  - Hard negatives: 1-2 impressed-but-not-clicked items
  - Random negatives: 1-2 uniformly sampled items (maintain calibration)
```

### 2.4 A Complete Training Record

After feature engineering (covered in Section 3), one training record looks like:

```
{
    # Identity
    "user_id": "u_382910",
    "video_id": "v_19283",
    "label": 1,
    "watch_time_sec": 187.5,
    "timestamp": 1700000000,

    # User features (precomputed)
    "user_features": {
        "user_id_emb":        [0.12, -0.34, ...],    # 64-d
        "age_bucket_emb":     [0.5, 0.1, ...],       # 8-d
        "country_emb":        [0.2, -0.1, ...],      # 16-d
        "language_emb":       [0.3, 0.7, ...],       # 8-d
        "device_onehot":      [0, 1, 0, 0],          # 4-d (mobile)
        "avg_watch_time_7d":  [2.14],                 # 1-d (log-transformed)
        "num_videos_7d":      [1.92],                 # 1-d (log-transformed)
        "watch_history_emb":  [0.05, 0.32, ...],     # 128-d (avg of last 50)
        "category_affinity":  [0.3, 0.1, 0.25, ...], # 20-d
        "hour_emb":           [0.4, -0.2, ...],      # 8-d
        "dow_emb":            [0.1, 0.5, ...],       # 4-d
    },
    # → concatenated: 262-d user feature vector

    # Item features (precomputed)
    "item_features": {
        "video_id_emb":       [-0.22, 0.45, ...],    # 64-d
        "creator_id_emb":     [0.11, -0.33, ...],    # 32-d
        "category_emb":       [0.6, 0.2, ...],       # 16-d
        "duration_bucket_emb":[0.3, -0.1, ...],      # 8-d
        "upload_age_days":    [1.61],                 # 1-d (log-transformed)
        "title_emb":          [0.08, -0.15, ...],    # 128-d (from BERT)
        "total_views_log":    [4.23],                 # 1-d
        "avg_completion_rate":[0.72],                 # 1-d
        "like_ratio":         [0.89],                 # 1-d
    },
    # → concatenated: 252-d item feature vector
}
```

---

## 3. Feature Engineering Deep Dive

This section shows exactly what raw data goes in, how each feature is processed, and what the model actually sees as input.

### 3.1 Raw Data: What We Have

Before any processing, here's what the databases and logs contain:

```
USER TABLE (from registration + profile service):
─────────────────────────────────────────────────
  user_id:          "u_382910"                (string, unique ID)
  signup_date:      "2022-06-15"              (date)
  country:          "US"                      (string, ~200 values)
  language:         "en"                      (string, ~50 values)
  age:              28                        (integer)
  gender:           "M"                       (string, M/F/Other)
  subscription:     "free"                    (string, free/premium)

VIDEO TABLE (from upload service + content pipeline):
─────────────────────────────────────────────────────
  video_id:         "v_19283"                 (string, unique ID)
  creator_id:       "c_5501"                  (string, FK to creator)
  title:            "How to Build a Transformer from Scratch"  (text)
  description:      "In this tutorial..."     (long text)
  category:         "Education"               (string, ~25 values)
  tags:             ["python", "AI", "tutorial"]  (list of strings)
  language:         "en"                      (string)
  duration_sec:     1440                      (integer, seconds)
  upload_time:      "2024-03-10T14:22:00Z"    (timestamp)
  thumbnail_url:    "https://..."             (image URL)

INTERACTION LOGS (from event stream, billions/day):
───────────────────────────────────────────────────
  user_id:          "u_382910"
  video_id:         "v_19283"
  event_type:       "watch"                   (impression/click/watch/like/share/dislike)
  timestamp:        1700000180
  watch_sec:        178                       (only for watch events)
  position:         3                         (slot position where video was shown)
  device:           "mobile_ios"              (string)

AGGREGATED STATS (precomputed daily by batch pipeline):
───────────────────────────────────────────────────────
  Per-user aggregates:
    avg_daily_watch_min:    45.2
    num_videos_watched_7d:  83
    num_videos_watched_30d: 312
    num_likes_7d:           12
    category_watch_counts:  {"Education": 35, "Music": 22, "Gaming": 15, ...}
    last_50_watched_ids:    ["v_19283", "v_10021", "v_88421", ...]

  Per-video aggregates:
    total_views:            1_234_567
    total_impressions:      8_500_000
    total_watch_time_sec:   890_000_000
    total_likes:            45_000
    total_dislikes:         1_200
    total_shares:           8_900
```

### 3.2 User Features: Raw → Processed → Model Input

Each raw feature needs to be converted into a fixed-size numeric vector the neural network can consume. Here's every user feature, showing the raw value, why it needs processing, and what the model actually sees.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        USER TOWER INPUT (262 dimensions)                        │
│                                                                                 │
│  Feature              Raw Value          Processing           Model Input  Dims │
│  ───────              ─────────          ──────────           ───────────  ──── │
│                                                                                 │
│  ── Identity ──                                                                 │
│  user_id              "u_382910"         hash → embedding     [0.12, -0.34, ..]  64│
│                                          lookup table                            │
│                                                                                 │
│  ── Demographics ──                                                             │
│  age                  28                 bucket → embedding   [0.5, 0.1, ..]     8│
│  country              "US"               lookup → embedding   [0.2, -0.1, ..]   16│
│  language             "en"               lookup → embedding   [0.3, 0.7, ..]     8│
│  gender               "M"               one-hot              [1, 0, 0]           3│
│  subscription         "free"             one-hot              [1, 0]              2│
│  device               "mobile_ios"       one-hot              [0, 1, 0, 0]       4│
│                                                                                 │
│  ── Behavioral (aggregated stats) ──                                            │
│  avg_daily_watch_min  45.2               log → z-score        [2.14]             1│
│  num_videos_7d        83                 log → z-score        [1.92]             1│
│  num_videos_30d       312                log → z-score        [2.45]             1│
│  num_likes_7d         12                 log → z-score        [1.08]             1│
│  account_age_days     612                log → z-score        [2.81]             1│
│                                                                                 │
│  ── Watch History (most important!) ──                                          │
│  last_50_watched_ids  ["v_19283", ...]   avg(video_embs)      [0.05, 0.32, ..]  128│
│                                                                                 │
│  ── Category Affinity ──                                                        │
│  category_watch_pcts  {Edu:0.35, ..}     normalize to pcts    [0.35, 0.22, ..]   24│
│                                          (one dim per category)                  │
│                                                                                 │
│  TOTAL                                                                     262  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Processing Each Feature Type in Detail

**Categorical → Embedding Lookup (most features)**

```python
# WHY: Categorical strings have no numeric meaning. "US" is not > "JP".
#      Embedding tables learn a dense vector per category that captures
#      similarity (e.g., US and CA might get similar embeddings).

# HOW:
#   1. Build a vocabulary: {"US": 0, "GB": 1, "JP": 2, ...}
#   2. Create an nn.Embedding table in the model
#   3. Look up the index at train/serve time

# Country: ~200 unique values → 16-d embedding
self.country_emb = nn.Embedding(num_countries, 16)
# country "US" → index 0 → self.country_emb(0) → [0.2, -0.1, ..] (16-d)

# Age: continuous but bucketized first for better generalization
#   Raw: 28 → bucket "25-34" → index 3 → embedding
age_buckets = [0, 13, 18, 25, 35, 45, 55, 65, 100]  # 8 buckets
self.age_emb = nn.Embedding(len(age_buckets), 8)

# Language: ~50 values → 8-d embedding
self.language_emb = nn.Embedding(num_languages, 8)
```

**High-Cardinality ID → Hash Embedding**

```python
# WHY: user_id has 100M+ unique values. A full embedding table would be
#      100M × 64 × 4 bytes = 25.6 GB — too large.

# HOW: Hash the ID into a fixed-size bucket, then look up.
#      Multiple users may collide into the same bucket (acceptable trade-off).

HASH_BUCKETS = 1_000_000  # 1M buckets instead of 100M
self.user_id_emb = nn.Embedding(HASH_BUCKETS, 64)

def get_user_id_emb(self, user_id_str):
    bucket = hash(user_id_str) % HASH_BUCKETS
    return self.user_id_emb(bucket)  # 64-d

# Alternative: multi-hash (reduces collision impact)
# Hash with 2 different seeds, average the embeddings
def get_user_id_emb_multi(self, user_id_str):
    bucket_a = hash_fn_a(user_id_str) % HASH_BUCKETS
    bucket_b = hash_fn_b(user_id_str) % HASH_BUCKETS
    return (self.user_id_emb_a(bucket_a) + self.user_id_emb_b(bucket_b)) / 2
```

**Low-Cardinality Categorical → One-Hot**

```python
# WHY: When there are only 2-5 values, embedding is overkill.
#      One-hot is simpler and the model can learn weights directly.

# Gender: M/F/Other → [1,0,0], [0,1,0], [0,0,1]
# Subscription: free/premium → [1,0], [0,1]
# Device: desktop/mobile_ios/mobile_android/tablet → [0,1,0,0]

gender_onehot = F.one_hot(torch.tensor(gender_idx), num_classes=3)  # 3-d
```

**Continuous Numeric → Log-Transform + Z-Score**

```python
# WHY: Raw values have extreme distributions (power-law).
#      avg_daily_watch_min: most users ~10 min, some ~600 min
#      Without log: gradients dominated by outliers, model ignores small values.

# HOW: log1p squishes the range, z-score centers at 0 with std 1.

# Step 1: Precompute stats on training data (offline, once)
log_watch_values = np.log1p(all_users["avg_daily_watch_min"])
MEAN_LOG_WATCH = log_watch_values.mean()    # e.g., 2.5
STD_LOG_WATCH = log_watch_values.std()      # e.g., 1.1

# Step 2: Apply at train and serve time
def process_continuous(value, mean, std):
    return (np.log1p(value) - mean) / (std + 1e-10)

# Example:
#   raw = 45.2 → log1p(45.2) = 3.83 → (3.83 - 2.5) / 1.1 = 1.21
#   raw = 3.0  → log1p(3.0)  = 1.39 → (1.39 - 2.5) / 1.1 = -1.01
#   raw = 600  → log1p(600)  = 6.40 → (6.40 - 2.5) / 1.1 = 3.55
#   Without log: [3.0, 45.2, 600] → huge variance
#   With log+z:  [-1.01, 1.21, 3.55] → manageable range
```

**Watch History → Average Pooling of Video Embeddings**

```python
# WHY: The most predictive feature. "What you watched" directly predicts
#      "what you'll watch next."  But it's a VARIABLE-LENGTH list of IDs.
#      The model needs a FIXED-SIZE vector.

# HOW: Look up each watched video's embedding, then average them.
#      The video embeddings come from a SHARED embedding table
#      (same table used in the item tower for video_id).

def compute_watch_history_embedding(watched_video_ids, video_emb_table, max_len=50):
    """
    watched_video_ids: ["v_19283", "v_10021", "v_88421", ...]  (last 50)
    video_emb_table: dict or nn.Embedding mapping video_id → 128-d vector

    Returns: 128-d vector summarizing user's watch history
    """
    # Take the most recent max_len videos
    recent = watched_video_ids[-max_len:]

    if len(recent) == 0:
        return np.zeros(128)  # Cold-start user: zero vector

    embs = np.array([video_emb_table[vid] for vid in recent])  # (L, 128)
    return embs.mean(axis=0)  # (128,)

# Three options for history encoding:
#
# Option A: Average pooling (YouTube 2016)
#   history_emb = mean(video_embs)
#   Pros: Simple, precomputable, O(1) at serving
#   Cons: Loses temporal order
#
# Option B: Time-weighted average
#   weight = 2^(-age_days / 7)  for each video
#   history_emb = weighted_mean(video_embs, weights)
#   Pros: Recent watches matter more
#   Cons: Needs timestamps, refresh more frequently
#
# Option C: Transformer self-attention (SASRec / TikTok)
#   history_emb = transformer(video_emb_sequence)[-1]
#   Pros: Captures sequential patterns ("Part 1 → Part 2")
#   Cons: Cannot precompute, expensive at serving time

# Comparison:
# ┌──────────────────┬──────────┬──────────────┬────────────────┐
# │ Method           │ Quality  │ Serving Cost │ Precomputable? │
# ├──────────────────┼──────────┼──────────────┼────────────────┤
# │ Average pooling  │ Good     │ O(1) lookup  │ Yes            │
# │ Time-weighted    │ Better   │ O(1) lookup  │ Yes (refresh)  │
# │ Transformer      │ Best     │ O(L×D)/req   │ No             │
# └──────────────────┴──────────┴──────────────┴────────────────┘
```

**Category Affinity → Normalized Distribution**

```python
# WHY: Shows the user's genre preferences as a probability distribution.

# Raw data:
#   category_watch_counts = {"Education": 35, "Music": 22, "Gaming": 15, "News": 8, ...}

# Processing: normalize to sum to 1.0
total = sum(category_watch_counts.values())  # 80
category_affinity = {k: v / total for k, v in category_watch_counts.items()}
# → {"Education": 0.44, "Music": 0.28, "Gaming": 0.19, "News": 0.10, ...}

# Model input: fixed-order vector (one dim per category, 24 categories → 24-d)
CATEGORY_ORDER = ["Education", "Music", "Gaming", "News", "Comedy", ...]  # 24 total
affinity_vector = [category_affinity.get(cat, 0.0) for cat in CATEGORY_ORDER]
# → [0.44, 0.28, 0.19, 0.10, 0.0, ...]  (24-d, sums to 1.0)
```

### 3.3 Video Features: Raw → Processed → Model Input

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        ITEM TOWER INPUT (252 dimensions)                        │
│                                                                                 │
│  Feature              Raw Value                Processing         Model   Dims  │
│  ───────              ─────────                ──────────         ─────   ────  │
│                                                                                 │
│  ── Identity ──                                                                 │
│  video_id             "v_19283"                hash → embedding   [..]     64   │
│  creator_id           "c_5501"                 hash → embedding   [..]     32   │
│                                                                                 │
│  ── Content Metadata ──                                                         │
│  category             "Education"              lookup → embedding [..]     16   │
│  language             "en"                     lookup → embedding [..]      8   │
│  duration_sec         1440                     bucket → embedding [..]      8   │
│  tags                 ["python","AI","tut.."]  multi-hot or embed [..]     16   │
│                                                                                 │
│  ── Text ──                                                                     │
│  title                "How to Build..."        sentence-BERT      [..]    128   │
│                                                (pretrained, frozen)              │
│                                                                                 │
│  ── Engagement Stats (aggregated, updated daily) ──                             │
│  total_views          1_234_567                log → z-score      [..]      1   │
│  avg_completion_rate  0.72                     raw (already 0-1)  [0.72]   1   │
│  like_ratio           0.974                    raw (already 0-1)  [0.974]  1   │
│  ctr                  0.045                    raw (already 0-1)  [0.045]  1   │
│                                                                                 │
│  ── Freshness ──                                                                │
│  upload_age_days      5                        log → z-score      [..]      1   │
│  hour_of_upload       14                       sin/cos cyclical   [..]      2   │
│                                                (sin(2π×14/24),                  │
│                                                 cos(2π×14/24))                  │
│                                                                                 │
│  IMPORTANT: All video features are the SAME regardless of which user is         │
│  viewing. Per-user signals are in the user tower.                               │
│                                                                                 │
│  TOTAL                                                                    252   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Processing Each Video Feature

**Video ID / Creator ID → Hash Embedding**

```python
# Same approach as user_id — too many unique values for a full table.
# video_id: ~10M+ videos,  creator_id: ~1M+ creators

VIDEO_HASH_BUCKETS = 500_000
CREATOR_HASH_BUCKETS = 200_000

self.video_id_emb = nn.Embedding(VIDEO_HASH_BUCKETS, 64)
self.creator_id_emb = nn.Embedding(CREATOR_HASH_BUCKETS, 32)

# WHY include video_id when we already have content features?
#   video_id embedding captures the COLLABORATIVE signal:
#     "users who watched this video also watched X"
#   Content features capture WHAT the video is about.
#   Together they give both collaborative + content-based signals.
```

**Title → Pretrained Sentence Embedding**

```python
# WHY: Title is free-form text. Can't one-hot encode millions of unique titles.
#      Pretrained language models understand semantic meaning.

# HOW: Run titles through a pretrained model ONCE (offline batch job).
#      Store the embeddings. Recompute only when titles change.

from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")  # produces 384-d

# Offline: precompute and store
title_emb_384 = encoder.encode("How to Build a Transformer from Scratch")

# In the item tower: a learned linear projection reduces 384-d → 128-d
self.title_proj = nn.Linear(384, 128)
# This projection is trained jointly with the rest of the model,
# so it learns which aspects of the title matter for recommendations.

# Alternatives:
# TF-IDF + SVD → 128-d  (simpler, loses semantic meaning)
# CLIP embedding         (captures title + thumbnail jointly)
```

**Duration → Bucketized Embedding**

```python
# WHY: Duration is continuous but has meaningful clusters:
#      shorts (<60s), short (1-5min), medium (5-20min), long (20-60min), movie (60min+)
#      Bucketizing lets the model learn different behaviors per length category.

duration_buckets = [0, 60, 300, 600, 1200, 3600, float('inf')]
# Bucket labels:     short  1-5m  5-10m  10-20m  20-60m   60m+

def bucketize_duration(duration_sec):
    for i, threshold in enumerate(duration_buckets[1:]):
        if duration_sec < threshold:
            return i
    return len(duration_buckets) - 2

self.duration_emb = nn.Embedding(len(duration_buckets) - 1, 8)

# 1440 sec → bucket 4 (20-60min) → self.duration_emb(4) → 8-d vector
```

**Tags → Multi-Hot or Pooled Embedding**

```python
# WHY: Tags are a variable-length set of strings.
#      Need to convert to fixed-size vector.

# Option A: Multi-hot (if tag vocabulary is small, < 500)
TAG_VOCAB = {"python": 0, "AI": 1, "tutorial": 2, "music": 3, ...}  # 200 tags
def tags_to_multihot(tags):
    vec = np.zeros(len(TAG_VOCAB))
    for tag in tags:
        if tag in TAG_VOCAB:
            vec[TAG_VOCAB[tag]] = 1.0
    return vec  # 200-d sparse binary vector
# Then reduce with a learned linear: nn.Linear(200, 16) → 16-d

# Option B: Average tag embeddings (if tag vocabulary is large)
self.tag_emb = nn.Embedding(num_tags, 16)
def tags_to_embedding(tag_indices):
    return self.tag_emb(tag_indices).mean(dim=0)  # 16-d
```

**Engagement Stats → Different Processing Per Stat**

```python
# Already-bounded values (0 to 1): use as-is
avg_completion_rate = 0.72   # → [0.72]   no processing needed
like_ratio          = 0.974  # → [0.974]  no processing needed
ctr                 = 0.045  # → [0.045]  no processing needed

# Unbounded counts: log + z-score (same as user continuous features)
total_views_processed = (np.log1p(1_234_567) - MEAN_LOG_VIEWS) / STD_LOG_VIEWS

# IMPORTANT: These are aggregated over ALL users, not per-user.
#   The item tower sees the SAME engagement stats for a video
#   regardless of who is viewing it.
```

**Upload Time → Cyclical Encoding for Hour, Log for Age**

```python
# Upload hour: cyclical (hour 23 is close to hour 0)
#   sin/cos encoding preserves this circular property.
hour = 14
hour_sin = np.sin(2 * np.pi * hour / 24)  # 0.866
hour_cos = np.cos(2 * np.pi * hour / 24)  # -0.5
# → [0.866, -0.5]  (2-d)

# Upload age: how old the video is (freshness signal)
upload_age_days = (now - upload_timestamp) / 86400  # e.g., 5 days
upload_age_processed = (np.log1p(upload_age_days) - MEAN_LOG_AGE) / STD_LOG_AGE
```

### 3.4 Concatenation: Building the Final Input Vectors

```python
def build_user_input(raw_user, user_history, video_emb_table, stats):
    """
    Transform raw user data into the 262-d vector the user tower expects.

    This runs both offline (training data construction)
    and online (serving, computing user embedding per request).
    """
    features = []

    # Identity: hash embedding lookup (done inside the model's forward pass)
    # Here we just store the hash index; the nn.Embedding lookup happens in the tower.
    user_id_hash = hash(raw_user["user_id"]) % HASH_BUCKETS  # → integer index

    # Demographics
    age_bucket = bucketize_age(raw_user["age"])        # → integer index
    country_idx = COUNTRY_VOCAB[raw_user["country"]]   # → integer index
    language_idx = LANG_VOCAB[raw_user["language"]]     # → integer index
    gender_onehot = one_hot(raw_user["gender"], 3)     # → [1, 0, 0]
    sub_onehot = one_hot(raw_user["subscription"], 2)  # → [1, 0]
    device_onehot = one_hot(raw_user["device"], 4)     # → [0, 1, 0, 0]

    # Behavioral (continuous → log + z-score)
    watch_min = log_zscore(raw_user["avg_daily_watch_min"], MEAN_W, STD_W)
    vids_7d = log_zscore(raw_user["num_videos_7d"], MEAN_V7, STD_V7)
    vids_30d = log_zscore(raw_user["num_videos_30d"], MEAN_V30, STD_V30)
    likes_7d = log_zscore(raw_user["num_likes_7d"], MEAN_L7, STD_L7)
    acct_age = log_zscore(raw_user["account_age_days"], MEAN_AA, STD_AA)

    # Watch history (variable-length list → fixed 128-d vector)
    history_emb = compute_watch_history_embedding(
        user_history["last_50_watched_ids"], video_emb_table
    )  # 128-d

    # Category affinity (dict → fixed 24-d vector)
    affinity = category_dict_to_vector(raw_user["category_watch_counts"])  # 24-d

    # The model's forward() method handles:
    #   1. Embedding lookups for user_id_hash, age_bucket, country_idx, language_idx
    #   2. Concatenation with the continuous/one-hot features above
    #   3. Feeding the 262-d vector through the tower MLP

    return {
        "id_indices": [user_id_hash, age_bucket, country_idx, language_idx],
        "continuous_and_onehot": np.concatenate([
            gender_onehot, sub_onehot, device_onehot,           # 3+2+4 = 9
            [watch_min, vids_7d, vids_30d, likes_7d, acct_age], # 5
            history_emb,                                         # 128
            affinity,                                            # 24
        ])  # 166-d (the remaining 96-d comes from embedding lookups in the model)
    }
```

```python
def build_video_input(raw_video, stats):
    """
    Transform raw video data into the 252-d vector the item tower expects.

    This runs OFFLINE as a batch job for ALL videos.
    The resulting embeddings are stored in the FAISS index.
    """
    video_id_hash = hash(raw_video["video_id"]) % VIDEO_HASH_BUCKETS
    creator_id_hash = hash(raw_video["creator_id"]) % CREATOR_HASH_BUCKETS
    category_idx = CATEGORY_VOCAB[raw_video["category"]]
    language_idx = LANG_VOCAB[raw_video["language"]]
    duration_bucket = bucketize_duration(raw_video["duration_sec"])

    tags_vec = tags_to_embedding(raw_video["tags"])    # 16-d
    title_emb = precomputed_title_embs[raw_video["video_id"]]  # 384-d → proj to 128

    # Engagement stats
    views_log = log_zscore(raw_video["total_views"], MEAN_VIEWS, STD_VIEWS)
    completion = raw_video["avg_completion_rate"]       # already 0-1
    like_ratio = raw_video["total_likes"] / (raw_video["total_likes"] + raw_video["total_dislikes"] + 1)
    ctr = raw_video["total_clicks"] / (raw_video["total_impressions"] + 1)

    # Freshness
    age_days = (now - raw_video["upload_time"]).days
    age_log = log_zscore(age_days, MEAN_AGE, STD_AGE)
    hour = raw_video["upload_time"].hour
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    return {
        "id_indices": [video_id_hash, creator_id_hash, category_idx, language_idx, duration_bucket],
        "continuous_and_encoded": np.concatenate([
            tags_vec,                                    # 16
            title_emb_128,                               # 128 (after projection)
            [views_log, completion, like_ratio, ctr],    # 4
            [age_log, hour_sin, hour_cos],               # 3
        ])  # 151-d (remaining 101-d from embedding lookups)
    }
```

### 3.5 Labels: How Positive and Negative Pairs Are Decided

The Two-Tower model is trained on **(user, video, label)** pairs. Defining what counts as positive vs. negative is one of the most important design decisions.

#### Positive Labels: "This user liked this video"

```
Raw interaction logs contain many event types.
We must choose WHICH events count as "positive."

Event Type          Use as Positive?    Reasoning
──────────          ────────────────    ─────────
Impression only     NO                  User saw it but didn't engage
Click only          WEAK positive       Might be clickbait
Watch < 30 sec      NO                  Likely accidental / bounced
Watch > 30 sec      YES (moderate)      Some real engagement
Watch > 50% dur     YES (strong)        Clearly interested
Liked               YES (strongest)     Explicit positive signal
Shared              YES (strongest)     Willing to endorse to others
Disliked            NO (explicit neg)   User rejected it

Recommended positive definition:
  positive = (watch_sec / duration > 0.5) OR (event == "like") OR (event == "share")

YouTube's approach (2016 paper):
  positive = completed watches, weighted by watch time in the loss
```

```python
def construct_positive_pairs(events_df):
    """
    From raw event logs → positive (user_id, video_id) pairs.

    Input: millions of raw events per day
    Output: millions of positive pairs for training

    Example:
      events for user u_42:
        impression v_100 (position 3)
        click v_100
        watch v_100 (178 sec / 240 sec = 74%) → POSITIVE (> 50%)
        impression v_201 (position 5)
        (no click)                             → NOT positive
        impression v_302 (position 7)
        click v_302
        watch v_302 (8 sec / 600 sec = 1.3%)  → NOT positive (< 50%)
        like v_100                             → POSITIVE (explicit)

      Result: [(u_42, v_100)]  (deduplicated)
    """
    # Meaningful watches
    watches = events_df[events_df["event"] == "watch"].copy()
    watches["completion"] = watches["watch_sec"] / watches["video_duration"]
    good_watches = watches[watches["completion"] > 0.5]

    # Explicit signals
    explicit_pos = events_df[events_df["event"].isin(["like", "share"])]

    # Combine and deduplicate
    positives = pd.concat([
        good_watches[["user_id", "video_id", "timestamp"]],
        explicit_pos[["user_id", "video_id", "timestamp"]],
    ]).drop_duplicates(subset=["user_id", "video_id"])

    return positives  # → each row is one positive training pair
```

#### Negative Labels: "This user would NOT like this video"

We only observe positive signals (watches, likes). The model also needs negatives. There are four strategies:

```
Strategy 1: RANDOM NEGATIVES
─────────────────────────────
  For user u_42 who watched [v_100, v_205]:
    Sample random videos they never saw: [v_88421, v_55012, v_91003]
    These become negative pairs: (u_42, v_88421, label=0), ...

  Pros: Simple, covers full item space
  Cons: Too easy — a cooking fan vs. a Korean drama is trivially distinguishable.
        Model doesn't learn fine-grained preferences.

Strategy 2: IN-BATCH NEGATIVES (industry standard)
───────────────────────────────────────────────────
  No explicit negative sampling needed!
  Within a batch of B positive pairs, each user's positive video
  becomes a negative for every OTHER user in the batch.

  Batch:                          Implicit labels:
    (u_42,  v_100) ← positive      u_42  vs v_100 = POSITIVE
    (u_78,  v_205) ← positive      u_42  vs v_205 = NEGATIVE (in-batch)
    (u_15,  v_317) ← positive      u_42  vs v_317 = NEGATIVE (in-batch)
    (u_91,  v_442) ← positive      u_42  vs v_442 = NEGATIVE (in-batch)

  With batch_size = 4096, each sample gets 4095 free negatives.

  Pros: Free, GPU-efficient (one matmul), scales with batch size
  Cons: Popular videos over-represented as negatives (popularity bias)
        → Fix with log-frequency correction (see Section 2.3)

Strategy 3: HARD NEGATIVES (impressed but not clicked)
──────────────────────────────────────────────────────
  Videos the user was SHOWN but did NOT click.
  These are the most informative: the system thought the user might like them.

    u_42 was shown: [v_100 (clicked), v_201 (skipped), v_302 (skipped)]
    Hard negatives: [(u_42, v_201, label=0), (u_42, v_302, label=0)]

  Pros: Forces fine-grained discrimination
  Cons: Selection bias (only includes videos the previous model chose)

Strategy 4: MIXED (best practice at YouTube/Google)
───────────────────────────────────────────────────
  Combine all three:
    In-batch negatives:  ~4095 per sample (free)
    Hard negatives:      1-2 per sample (from impression logs)
    Random negatives:    1-2 per sample (calibration)
```

#### Complete Training Record Example

Putting it all together — one training record that the model actually sees:

```
ONE TRAINING RECORD (after all feature processing):
────────────────────────────────────────────────────

user_id: "u_382910"   video_id: "v_19283"   label: POSITIVE
(because u_382910 watched 74% of v_19283)

USER TOWER INPUT (262-d):
  user_id hash idx  → [embedding lookup → 64-d]
  age bucket idx    → [embedding lookup → 8-d]
  country idx       → [embedding lookup → 16-d]
  language idx      → [embedding lookup → 8-d]
  gender            → [1, 0, 0]                        (3-d one-hot)
  subscription      → [1, 0]                           (2-d one-hot)
  device            → [0, 1, 0, 0]                     (4-d one-hot)
  avg_watch_min     → [1.21]                           (1-d log+zscore)
  num_vids_7d       → [1.92]                           (1-d log+zscore)
  num_vids_30d      → [2.45]                           (1-d log+zscore)
  num_likes_7d      → [1.08]                           (1-d log+zscore)
  acct_age_days     → [2.81]                           (1-d log+zscore)
  watch_history_emb → [0.05, 0.32, -0.11, ...]        (128-d avg pooling)
  category_affinity → [0.35, 0.22, 0.19, 0.10, ...]   (24-d normalized)
                                                        ──────
                                                        262-d total

ITEM TOWER INPUT (252-d):
  video_id hash idx   → [embedding lookup → 64-d]
  creator_id hash idx → [embedding lookup → 32-d]
  category idx        → [embedding lookup → 16-d]
  language idx        → [embedding lookup → 8-d]
  duration bucket idx → [embedding lookup → 8-d]
  tags_emb            → [0.4, -0.2, 0.1, ...]         (16-d)
  title_emb           → [0.08, -0.15, 0.22, ...]      (128-d sentence-BERT → projected)
  total_views         → [1.83]                         (1-d log+zscore)
  completion_rate     → [0.72]                         (1-d, already bounded)
  like_ratio          → [0.974]                        (1-d, already bounded)
  ctr                 → [0.045]                        (1-d, already bounded)
  upload_age_days     → [-0.52]                        (1-d log+zscore)
  upload_hour         → [0.866, -0.5]                  (2-d sin/cos)
                                                        ──────
                                                        252-d total

WHAT HAPPENS DURING TRAINING:
  user_tower(262-d)  → user_emb (128-d, L2-normalized)
  item_tower(252-d)  → item_emb (128-d, L2-normalized)
  dot(user_emb, item_emb) / temperature → score
  In-batch softmax loss pushes this score HIGH (it's a positive pair)
  and pushes dot(u_382910_emb, other_video_embs) LOW (in-batch negatives)
```

### 3.6 Feature Processing Cheat Sheet

```
┌───────────────────────┬──────────────────┬─────────────────────────────────────┐
│ Raw Data Type         │ Processing       │ Why This Processing                 │
├───────────────────────┼──────────────────┼─────────────────────────────────────┤
│ High-cardinality ID   │ Hash → embedding │ Too many values for full table      │
│ (user_id, video_id)   │                  │ Hash reduces memory, embedding      │
│                       │                  │ learns dense representation         │
├───────────────────────┼──────────────────┼─────────────────────────────────────┤
│ Medium-cardinality    │ Vocab → embedding│ 50-500 values: each gets a learned  │
│ (country, category)   │                  │ vector that captures similarity     │
├───────────────────────┼──────────────────┼─────────────────────────────────────┤
│ Low-cardinality       │ One-hot          │ 2-5 values: simple, direct weights  │
│ (gender, device)      │                  │                                     │
├───────────────────────┼──────────────────┼─────────────────────────────────────┤
│ Continuous (unbounded)│ log1p → z-score  │ Power-law data has extreme outliers │
│ (views, watch time)   │                  │ Log compresses, z-score centers     │
├───────────────────────┼──────────────────┼─────────────────────────────────────┤
│ Continuous (bounded)  │ Use as-is        │ Already in [0,1], no transform      │
│ (completion rate, CTR)│                  │ needed                              │
├───────────────────────┼──────────────────┼─────────────────────────────────────┤
│ Continuous (bucketed) │ Bucket → embed   │ Non-linear relationship with target │
│ (age, duration)       │                  │ Buckets learn per-range behavior    │
├───────────────────────┼──────────────────┼─────────────────────────────────────┤
│ Text (title)          │ Pretrained model │ Semantic meaning preserved          │
│                       │ → project dim    │ Fine-tuned via projection layer     │
├───────────────────────┼──────────────────┼─────────────────────────────────────┤
│ Variable-length list  │ Average pooling  │ Need fixed-size vector from list    │
│ (watch history, tags) │ (or transformer) │ Avg is simple; transformer is best  │
├───────────────────────┼──────────────────┼─────────────────────────────────────┤
│ Cyclical (hour, day)  │ sin/cos encoding │ Hour 23 must be near hour 0         │
│                       │                  │ sin/cos preserves circular distance │
├───────────────────────┼──────────────────┼─────────────────────────────────────┤
│ Dict/distribution     │ Normalize to     │ Fixed-size vector from variable     │
│ (category affinity)   │ probability vec  │ key-value pairs                     │
└───────────────────────┴──────────────────┴─────────────────────────────────────┘
```

### 3.7 Feature Pipeline Architecture

```
                    Offline (batch, daily/hourly)               Online (real-time)
                    ────────────────────────────                ──────────────────

Raw events ──→ Spark / Flink ──→ Feature Store (Hive / BigQuery)
    │                                    │
    │                                    │ Precomputed features, refreshed on schedule:
    │                                    │
    │                                    ├── user_id hash idx        ──→ Redis (user features)
    │                                    ├── watch_history_emb (128d) ──→ Redis (hourly refresh)
    │                                    ├── category_affinity (24d)  ──→ Redis (daily refresh)
    │                                    ├── user aggregated stats    ──→ Redis (daily refresh)
    │                                    │
    │                                    ├── video_id hash idx        ──→ Redis (item features)
    │                                    ├── title_emb (128d)         ──→ Redis (on upload)
    │                                    ├── video engagement stats   ──→ Redis (daily refresh)
    │                                    └── tags_emb (16d)           ──→ Redis (on upload)
    │
    │                    Training (offline)
    │                    ──────────────────
    └──→ Join user feats + video feats + labels ──→ Parquet ──→ DataLoader ──→ GPU

Feature freshness tiers:
  Real-time (computed per request): device_type, hour_of_day, day_of_week
  Near real-time (streaming, minutes): append latest watch to history
  Hourly batch: watch_history_emb, category_affinity
  Daily batch: engagement stats (views, completion rate, like ratio)
  Static (until retrained): ID embeddings, title_emb, tags_emb
```

---

## 4. Loss Functions: Three Options

### Option 1: Softmax Cross-Entropy with In-Batch Negatives (Recommended)

Treat retrieval as a multi-class classification: "which item in the batch does this user match?"

```python
class InBatchSoftmaxLoss(nn.Module):
    """
    In-batch negative softmax loss.

    For a batch of B (user, item) pairs:
    - Compute B×B similarity matrix
    - Diagonal = positive pairs
    - Off-diagonal = negative pairs
    - Apply softmax cross-entropy

    Used by: YouTube, Google, Facebook
    """

    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, user_embs, item_embs):
        # user_embs: (B, D), item_embs: (B, D), both L2-normalized

        logits = torch.matmul(user_embs, item_embs.T) / self.temperature  # (B, B)
        labels = torch.arange(logits.size(0), device=logits.device)

        return F.cross_entropy(logits, labels)
```

**Temperature parameter**: Controls how "sharp" the similarity distribution is.
- `temperature = 1.0`: Soft distribution, model is tolerant of similar-score items
- `temperature = 0.05`: Sharp distribution, model must clearly separate positive from negatives
- **Lower temperature → harder training but better discrimination** (typical: 0.05 to 0.1)

### Option 2: Triplet Loss

Explicitly push positive closer and negative farther.

```python
class TripletLoss(nn.Module):
    """
    Triplet loss: ensure positive is closer than negative by a margin.

    loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)

    Requires explicit negative sampling (can't use in-batch trick as easily).
    """

    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, user_embs, pos_item_embs, neg_item_embs):
        # All (B, D), L2-normalized

        pos_sim = (user_embs * pos_item_embs).sum(dim=1)   # (B,)
        neg_sim = (user_embs * neg_item_embs).sum(dim=1)   # (B,)

        loss = F.relu(neg_sim - pos_sim + self.margin)
        return loss.mean()
```

**Pros**: Intuitive, works with explicit negative sampling.
**Cons**: Requires mining good triplets; easy triplets contribute zero loss (wasted computation). Less GPU-efficient than in-batch softmax.

### Option 3: Binary Cross-Entropy (Pointwise)

Treat each (user, item) pair independently as a binary classification.

```python
class PointwiseBCELoss(nn.Module):
    """
    Pointwise binary cross-entropy.

    Each (user, item, label) is an independent sample.
    Requires explicit negative sampling.
    """

    def forward(self, user_embs, item_embs, labels):
        # user_embs, item_embs: (B, D), labels: (B,) in {0, 1}

        scores = (user_embs * item_embs).sum(dim=1)    # (B,)
        probs = torch.sigmoid(scores)

        loss = F.binary_cross_entropy(probs, labels.float())
        return loss
```

**Pros**: Simple, each sample is independent.
**Cons**: Need to manage positive/negative ratio carefully. Doesn't benefit from in-batch negatives as naturally.

### Loss Function Comparison

| Loss | Negatives | GPU Efficiency | Quality | Used By |
|------|-----------|---------------|---------|---------|
| **Softmax + in-batch** | Free (from batch) | Best (one matmul) | Best | YouTube, Google, CLIP |
| Triplet | Explicit sampling | Medium | Good | Older systems |
| Binary CE | Explicit sampling | Low | Adequate | Simple baselines |

**Recommendation**: Softmax with in-batch negatives. It's the industry standard for a reason — free negatives, GPU-efficient, and strong empirical performance.

---

## 5. Complete Training Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


# ──────────────────────────────────────────────
# Model Definition
# ──────────────────────────────────────────────

class Tower(nn.Module):
    """A single tower (used for both user and item)."""

    def __init__(self, input_dim, hidden_dims=[512, 256], output_dim=128, dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        emb = self.net(x)
        return F.normalize(emb, p=2, dim=1)  # L2 normalize → unit sphere


class TwoTowerModel(nn.Module):
    """
    Two-Tower retrieval model.

    Architecture:
        User features → User Tower → user_emb (128-d, L2-normalized)
        Item features → Item Tower → item_emb (128-d, L2-normalized)
        Score = dot(user_emb, item_emb) / temperature
    """

    def __init__(self, user_feat_dim, item_feat_dim, embedding_dim=128, temperature=0.05):
        super().__init__()
        self.user_tower = Tower(user_feat_dim, output_dim=embedding_dim)
        self.item_tower = Tower(item_feat_dim, output_dim=embedding_dim)
        self.temperature = temperature

    def forward(self, user_features, item_features):
        user_emb = self.user_tower(user_features)     # (B, D)
        item_emb = self.item_tower(item_features)     # (B, D)

        # In-batch negatives: B×B similarity matrix
        logits = torch.matmul(user_emb, item_emb.T) / self.temperature

        return {"user_emb": user_emb, "item_emb": item_emb, "logits": logits}

    def compute_user_emb(self, user_features):
        """Inference: compute user embedding only."""
        return self.user_tower(user_features)

    def compute_item_emb(self, item_features):
        """Inference: compute item embedding only (for precomputation)."""
        return self.item_tower(item_features)


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class InteractionDataset(Dataset):
    """
    Dataset of (user_features, item_features) positive pairs.

    Each row is a positive interaction: user watched/liked a video.
    Negatives are handled via in-batch negative sampling during training,
    so the dataset only contains positive pairs.
    """

    def __init__(self, user_features, item_features):
        """
        Args:
            user_features: np.ndarray of shape (N, user_feat_dim)
            item_features: np.ndarray of shape (N, item_feat_dim)
        """
        assert len(user_features) == len(item_features)
        self.user_features = torch.FloatTensor(user_features)
        self.item_features = torch.FloatTensor(item_features)

    def __len__(self):
        return len(self.user_features)

    def __getitem__(self, idx):
        return {
            "user_features": self.user_features[idx],
            "item_features": self.item_features[idx],
        }


# ──────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────

def train(model, train_loader, val_loader, device, epochs=10, lr=1e-3):
    """
    Full training loop with in-batch negative loss.

    Key details:
    - Large batch size is critical (2048-8192) for enough in-batch negatives
    - Learning rate warmup helps stability
    - Temperature is usually fixed, not learned
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.to(device)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            user_feat = batch["user_features"].to(device)
            item_feat = batch["item_features"].to(device)

            output = model(user_feat, item_feat)

            # In-batch softmax loss
            labels = torch.arange(user_feat.size(0), device=device)
            loss = F.cross_entropy(output["logits"], labels)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (important for stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / num_batches

        # ── Validate ──
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                user_feat = batch["user_features"].to(device)
                item_feat = batch["item_features"].to(device)

                output = model(user_feat, item_feat)
                labels = torch.arange(user_feat.size(0), device=device)
                loss = F.cross_entropy(output["logits"], labels)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches

        print(f"Epoch {epoch+1}/{epochs}: "
              f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, "
              f"lr={scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_two_tower.pt")
            print(f"  → Saved best model (val_loss={best_val_loss:.4f})")

    return model


# ──────────────────────────────────────────────
# Usage Example
# ──────────────────────────────────────────────

"""
# Dimensions from our feature engineering (Section 3 of README)
USER_FEAT_DIM = 262    # user features concatenated
ITEM_FEAT_DIM = 252    # item features concatenated
EMBEDDING_DIM = 128
BATCH_SIZE = 4096      # Large batch → more in-batch negatives

# Load data
train_user_feats = np.load("train_user_features.npy")   # (N_train, 262)
train_item_feats = np.load("train_item_features.npy")   # (N_train, 252)
val_user_feats = np.load("val_user_features.npy")
val_item_feats = np.load("val_item_features.npy")

train_dataset = InteractionDataset(train_user_feats, train_item_feats)
val_dataset = InteractionDataset(val_user_feats, val_item_feats)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=8, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, drop_last=True)

# Note: drop_last=True is important — last batch may be small,
# giving too few in-batch negatives

model = TwoTowerModel(USER_FEAT_DIM, ITEM_FEAT_DIM, EMBEDDING_DIM, temperature=0.05)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train(model, train_loader, val_loader, device, epochs=10, lr=1e-3)
"""
```

---

## 6. After Training: Building the ANN Index

Once the model is trained, the two towers are used **separately**:

```
                    Training (joint)
                    ────────────────
    User Tower ←──── backprop ────→ Item Tower
         ↑                              ↑
    user_features                  item_features

                    Serving (separate)
                    ──────────────────

    OFFLINE (batch job, every 4-6 hours):
    ─────────────────────────────────────
    For ALL 10M videos:
      item_features → Item Tower → item_emb (128-d)
      Store in FAISS index

    ONLINE (per request, ~5ms):
    ───────────────────────────
    For THIS user:
      user_features → User Tower → user_emb (128-d)
      FAISS search(user_emb, k=100) → top-100 candidate video IDs
```

```python
import faiss

def build_ann_index(model, item_dataloader, device):
    """
    Precompute all item embeddings and build FAISS index.

    Run as a batch job every 4-6 hours.
    """
    model.eval()
    all_embeddings = []
    all_video_ids = []

    with torch.no_grad():
        for batch in item_dataloader:
            item_feat = batch["item_features"].to(device)
            item_emb = model.compute_item_emb(item_feat)   # Only item tower
            all_embeddings.append(item_emb.cpu().numpy())
            all_video_ids.extend(batch["video_id"])

    embeddings = np.vstack(all_embeddings).astype("float32")

    # Choose index type based on corpus size
    dim = embeddings.shape[0]
    n_items = len(embeddings)

    if n_items < 100_000:
        # Small corpus: exact search is fine
        index = faiss.IndexFlatIP(dim)
    elif n_items < 10_000_000:
        # Medium corpus: IVF-Flat
        nlist = int(np.sqrt(n_items))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings[:min(500_000, n_items)])
        index.nprobe = 32
    else:
        # Large corpus: IVF-PQ for memory efficiency
        nlist = 4096
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, 32, 8)
        index.train(embeddings[:500_000])
        index.nprobe = 64

    index.add(embeddings)

    print(f"Built ANN index: {index.ntotal} items")
    return index, all_video_ids


def serve_user_request(model, user_features, index, video_ids, device, top_k=100):
    """
    Online serving: compute user embedding, ANN search.

    Latency budget: < 10ms total
      - User tower forward pass: ~2ms
      - FAISS search: ~5ms
      - Result packaging: ~1ms
    """
    model.eval()
    with torch.no_grad():
        user_emb = model.compute_user_emb(user_features.to(device))  # Only user tower
        user_emb_np = user_emb.cpu().numpy().astype("float32")

    scores, indices = index.search(user_emb_np, top_k)

    candidates = [
        {"video_id": video_ids[idx], "score": float(score)}
        for idx, score in zip(indices[0], scores[0])
        if idx >= 0
    ]
    return candidates
```

---

## 7. Training Infrastructure & Scale

### 7.1 Data Scale

```
YouTube-scale:
  Raw events/day:      ~100B impressions, ~10B watches
  Training pairs/day:  ~1-5B positive pairs
  Training data size:  ~1-10 TB per day (features included)

Typical startup:
  Training pairs:      ~10-100M
  Training data:       ~10-100 GB

Training frequency:
  Full retrain:        Weekly (on last 30-90 days of data)
  Incremental update:  Daily (fine-tune on last day's data)
```

### 7.2 Training Time

| Scale | Hardware | Training Time |
|-------|----------|---------------|
| 10M pairs | 1× A100 GPU | ~30 min |
| 100M pairs | 4× A100 GPU | ~2 hours |
| 1B pairs | 8× A100 GPU (DDP) | ~6 hours |
| 10B pairs | 32× GPU (multi-node) | ~12-24 hours |

### 7.3 Distributed Training

```python
# PyTorch DistributedDataParallel (DDP) for multi-GPU training

# Key challenge with in-batch negatives and DDP:
# Each GPU has its own batch → negatives are only within that GPU's batch.
# Fix: all-gather embeddings across GPUs before computing the similarity matrix.

def distributed_in_batch_loss(user_embs, item_embs, temperature):
    """
    Gather embeddings from all GPUs to create a larger negative pool.

    4 GPUs × batch_size 4096 = 16384 effective negatives per sample.
    """
    # Gather from all GPUs
    all_user_embs = torch.cat(torch.distributed.nn.all_gather(user_embs), dim=0)
    all_item_embs = torch.cat(torch.distributed.nn.all_gather(item_embs), dim=0)

    # Compute similarity matrix with ALL gathered embeddings
    logits = torch.matmul(all_user_embs, all_item_embs.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)

    return F.cross_entropy(logits, labels)
```

### 7.4 Training Data Split

```
CRITICAL: Use temporal split, NOT random split.

Random split (WRONG):
  Randomly assign 80/10/10 of interactions to train/val/test
  → Future interactions leak into training → overly optimistic metrics

Temporal split (CORRECT):
  Train:  Day 1 to Day 28   (all interactions)
  Val:    Day 29 to Day 30   (predict these from model trained on Day 1-28)
  Test:   Day 31 to Day 33   (final evaluation)

Timeline:
  Day 1        Day 28    Day 30    Day 33
  |──── Train ────|── Val ──|── Test ──|
```

---

## 8. Training Challenges & Solutions

### 8.1 Popularity Bias

**Problem**: Popular videos dominate as in-batch negatives. The model learns to demote them, causing under-recommendation of genuinely popular content.

```
Batch of 4096 samples:
  "Despacito" (10B views) appears as a negative ~50 times
  "Obscure podcast ep. 47" appears as a negative ~0 times

Model learns: "Despacito" = negative → score it low
But some users genuinely want Despacito!
```

**Solution**: Log-frequency correction (YouTube 2019).

```python
# Subtract log(item_frequency) from logits
# Frequently-sampled negatives get their penalty reduced
correction = torch.log(item_freq_in_batch + 1e-10)
logits = logits - correction.unsqueeze(0)
```

### 8.2 Folding — Embedding Collapse

**Problem**: All embeddings converge to a small region of the space. High similarity everywhere, model can't distinguish items.

```
Before folding: embeddings spread across the unit sphere
After folding:  all embeddings cluster in one region → all scores ≈ 0.95
```

**Solutions**:
- L2 normalization (already included) prevents magnitude collapse
- Temperature scaling keeps gradients informative
- BatchNorm in the towers maintains feature spread
- Monitor average pairwise similarity — if it approaches 1.0, something is wrong

### 8.3 Cold Start

**Problem**: New users have no watch history. New videos have no engagement stats.

```
New user: watch_history_emb = zeros(128), category_affinity = uniform, user_id_emb = random

New video: total_views_log = 0, avg_completion_rate = 0, like_ratio = 0
           But title_emb and category_emb are available!
```

**Solutions**:
- **New users**: Fall back to popularity/trending retrieval. As soon as they watch 2-3 videos, the watch_history_emb becomes meaningful.
- **New videos**: The item tower can still compute a reasonable embedding from metadata (title, category, creator, duration). The ID embedding will be random initially, so consider reducing its weight for new items.
- **Feature dropout during training**: Randomly zero out watch_history_emb with 10% probability during training. This forces the model to learn to be useful even without history.

```python
def forward(self, user_features, training=False):
    if training and random.random() < 0.1:
        # Zero out history embedding to simulate cold-start users
        user_features[:, history_start:history_end] = 0
    return self.user_tower(user_features)
```

### 8.4 Training-Serving Skew

**Problem**: Features computed differently in training (batch, from logged data) vs. serving (real-time, current data).

```
Training: watch_history_emb = average of 50 most recent watches at EVENT TIME
Serving:  watch_history_emb = average of 50 most recent watches at REQUEST TIME

If the feature store computes it differently (different code path,
different rounding, different video_emb version), scores shift.
```

**Solutions**:
- Log features at serving time and use those SAME logged features for training (YouTube's approach)
- Run feature validation: compare training-time and serving-time feature distributions
- Monitor model score distributions in production — sudden shifts indicate skew

---

## 9. Hyperparameter Guide

| Hyperparameter | Typical Range | Impact | Guidance |
|---------------|--------------|--------|----------|
| **embedding_dim** | 64-256 | Quality vs. memory/speed | 128 is the standard choice |
| **temperature** | 0.01-0.2 | Sharpness of similarity | 0.05-0.1 is common; lower = harder training, better discrimination |
| **batch_size** | 2048-8192 | Number of in-batch negatives | Larger is better (more negatives); limited by GPU memory |
| **learning_rate** | 1e-4 to 1e-3 | Training speed vs. stability | 1e-3 with cosine decay is safe |
| **hidden_dims** | [512,256] or [1024,512,256] | Model capacity | Deeper isn't always better; 2-3 layers is typical |
| **dropout** | 0.1-0.3 | Regularization | 0.2 is a safe default |
| **history_length** | 20-100 | How much user context | 50 is common; more helps but costs memory |
| **weight_decay** | 1e-5 to 1e-4 | L2 regularization | Prevents ID embeddings from growing unbounded |

---

## 10. Evaluation After Training

### 10.1 Offline Metrics

```python
def evaluate_retrieval(model, user_loader, index, video_ids, ground_truth,
                       device, k_values=[10, 50, 100, 200]):
    """
    Evaluate retrieval quality.

    ground_truth: dict of {user_id: [list of video_ids they'll actually watch]}
    """
    model.eval()
    results = {k: [] for k in k_values}

    with torch.no_grad():
        for batch in user_loader:
            user_feat = batch["user_features"].to(device)
            user_ids = batch["user_id"]

            user_embs = model.compute_user_emb(user_feat).cpu().numpy()

            for i, uid in enumerate(user_ids):
                if uid not in ground_truth:
                    continue

                relevant = set(ground_truth[uid])
                scores, indices = index.search(
                    user_embs[i:i+1].astype("float32"), max(k_values)
                )

                retrieved = [video_ids[idx] for idx in indices[0] if idx >= 0]

                for k in k_values:
                    top_k_set = set(retrieved[:k])
                    recall = len(top_k_set & relevant) / len(relevant)
                    results[k].append(recall)

    for k in k_values:
        avg_recall = np.mean(results[k])
        print(f"Recall@{k}: {avg_recall:.4f}")

    return results
```

### 10.2 What Good Looks Like

| Metric | Poor | Acceptable | Good | Great |
|--------|------|-----------|------|-------|
| Recall@100 | < 0.10 | 0.10-0.20 | 0.20-0.35 | > 0.35 |
| Recall@200 | < 0.15 | 0.15-0.30 | 0.30-0.45 | > 0.45 |
| HitRate@100 | < 0.40 | 0.40-0.60 | 0.60-0.75 | > 0.75 |

Note: These numbers may seem low, but retrieval only needs to put relevant items *somewhere* in the top-100 — the ranking model will sort them. A Recall@100 of 0.30 means 30% of what the user will actually watch in the next session appears in your 100 candidates. Given there are millions of videos, that's strong performance.

---

## 11. Training Options Comparison

| Decision | Option A | Option B | Recommendation |
|----------|----------|----------|----------------|
| **Loss function** | Softmax + in-batch negatives | Triplet loss | Softmax — more GPU-efficient, better empirical results |
| **Negative sampling** | In-batch only | In-batch + hard negatives | Start with in-batch only; add hard negatives if quality plateaus |
| **History encoding** | Average pooling | Transformer | Average pooling first; Transformer if serving latency allows |
| **Embedding dim** | 64 (faster) | 256 (richer) | 128 — best balance of quality and serving speed |
| **Temperature** | 0.1 (softer) | 0.03 (sharper) | 0.05 — good default, tune based on validation |
| **Training data window** | 7 days | 90 days | 30 days — captures trends without too much staleness |
| **Popularity correction** | None | Log-frequency correction | Add correction — prevents popular item suppression |
| **ID embeddings** | Full vocabulary | Hash embeddings | Full if < 10M items; hash if > 10M |

---

## 12. Interview Talking Points

### "Walk me through how Two-Tower training works."

> We collect positive pairs from user interaction logs — typically completed watches. We train two neural networks jointly: a user tower and an item tower. The key is the in-batch negative softmax loss: within a batch of B positive pairs, we compute a B×B similarity matrix where the diagonal contains positive pairs and off-diagonal entries are in-batch negatives. Cross-entropy loss pushes positive similarity up and negative similarity down. Gradients flow through both towers, forcing them to produce embeddings in a shared space.

### "Why not train two separate models?"

> They must learn to speak the same language. If trained separately, user embeddings might encode "likes gaming" in dimension 5, while item embeddings encode "is a game" in dimension 47. Joint training with the dot product objective forces alignment — both towers must agree on what each dimension means.

### "What's the most important hyperparameter?"

> Batch size. In-batch negatives scale with batch size — a batch of 4096 gives 4095 negatives per sample. Too small a batch (e.g., 64) means the model sees very few negatives per update, leading to poor discrimination. We use 4096-8192 with distributed all-gather across GPUs for even more negatives.

### "How do you handle the cold-start problem?"

> Two approaches: (1) For new users, fall back to popularity-based retrieval until they accumulate enough history. We also use feature dropout during training — randomly zero out watch history with 10% probability — so the model learns to use demographic and contextual features when history is unavailable. (2) For new videos, the item tower can compute embeddings from metadata alone (title, category, duration) without needing interaction statistics. The video_id embedding starts random but other features carry the load.

### "What are the biggest training pitfalls?"

> Three main ones: (1) Popularity bias — popular items dominate as in-batch negatives, causing the model to unfairly suppress them. Fix with log-frequency correction. (2) Training-serving skew — if features are computed differently at training time vs. serving time, model performance degrades silently. Fix by logging serving-time features and reusing them for training. (3) Embedding collapse — all embeddings converge to a small region. Monitor average pairwise cosine similarity; it should be near 0 for random pairs, not 0.9+.

### "How is this related to matrix factorization?"

> Two-Tower is a generalization of matrix factorization. MF learns `user_emb = Embedding[user_id]` and `item_emb = Embedding[item_id]`, with score = dot product. Two-Tower replaces the simple embedding lookup with a neural network that takes *multiple features* (ID + demographics + history + metadata) and applies non-linear transformations. If you strip away all features except IDs and use a single linear layer, Two-Tower degenerates into MF. (See `two_tower_vs_matrix_factorization.md` for details.)
