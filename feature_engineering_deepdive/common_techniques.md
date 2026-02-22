# Feature Engineering: Complete Guide

How to transform raw data into model inputs. This is often the highest-leverage work in ML — better features improve model quality more than better architectures.

---

## 1. The Big Picture

Every piece of raw data must become a **fixed-length numeric vector** before a model can use it. Feature engineering is the process of deciding how.

```
Raw data                          Feature engineering              Model input
─────────                         ───────────────────              ───────────
"gaming"                    →     embedding lookup          →     [0.8, 0.1, -0.2, ...]  (16-d)
"2024-03-15 20:30:00"       →     extract hour, cyclical    →     [sin(20h), cos(20h)]   (2-d)
1,250,000 views             →     log transform + normalize →     [2.14]                 (1-d)
"How to Build a Transformer"→     sentence encoder          →     [0.05, 0.32, ...]      (128-d)
[v1, v2, v3, ..., v50]      →     average embeddings        →     [0.11, -0.08, ...]     (128-d)
```

---

## 2. Feature Types and How to Process Each

### 2.1 Numeric / Continuous Features

Raw numbers: view counts, watch time, prices, ages, distances.

#### Technique 1: Log Transform

For features with **heavy right skew** (power-law distribution): views, follower counts, revenue.

```python
import numpy as np

# Problem: raw values span huge range
views = [10, 50, 200, 1000, 50000, 10000000]
# Gradient dominated by outliers, model struggles with scale

# Solution: log transform compresses the range
log_views = np.log1p(views)  # log1p = log(1+x), handles x=0
# [2.4, 3.9, 5.3, 6.9, 10.8, 16.1]
# Now the range is manageable and differences at the low end are preserved
```

**When to use**: Any feature where the distribution has a long tail — view counts, watch time, follower counts, revenue, time durations, population counts.

**When NOT to use**: Features already uniformly distributed (e.g., percentages, ratios).

#### Technique 2: Z-Score Normalization (Standardization)

Center the feature to mean=0 and std=1.

```python
from sklearn.preprocessing import StandardScaler

# z = (x - mean) / std
scaler = StandardScaler()
normalized = scaler.fit_transform(values.reshape(-1, 1))

# Often combined with log:
# Step 1: log transform (fix skew)
# Step 2: z-score normalize (fix scale)
log_normalized = (np.log1p(views) - mean_log) / std_log
```

**When to use**: Neural networks (sensitive to feature scale). Less important for tree models (XGBoost/LightGBM are scale-invariant).

#### Technique 3: Min-Max Scaling

Scale to [0, 1] range.

```python
from sklearn.preprocessing import MinMaxScaler

# x_scaled = (x - min) / (max - min)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values.reshape(-1, 1))
```

**When to use**: When you need bounded outputs (e.g., feeding into sigmoid). Sensitive to outliers — a single extreme value compresses everything else near 0.

#### Technique 4: Bucketizing / Binning

Convert continuous values into discrete buckets.

```python
# Video duration → categorical buckets
def bucketize_duration(seconds):
    if seconds < 60:      return "short"       # < 1 min
    elif seconds < 300:   return "medium"      # 1-5 min
    elif seconds < 900:   return "long"        # 5-15 min
    elif seconds < 1800:  return "very_long"   # 15-30 min
    else:                 return "extra_long"  # 30+ min

# Then treat as categorical → embedding or one-hot
```

**When to use**: When the relationship is non-linear and step-wise (e.g., "short videos behave fundamentally differently from long ones"). Also useful for tree models to reduce noise.

#### Technique 5: Clipping

Cap extreme values to prevent outliers from dominating.

```python
# Clip watch time to [0, 3600] seconds (cap at 1 hour)
watch_time_clipped = np.clip(watch_time, 0, 3600)

# Percentile-based clipping (more robust)
p1, p99 = np.percentile(values, [1, 99])
clipped = np.clip(values, p1, p99)
```

#### Summary: Numeric Feature Processing Decision Tree

```
Is it a count/amount with heavy tail?
├── Yes → log1p transform first
│         Then normalize (z-score for neural nets)
│
├── Is it a ratio or percentage (already bounded)?
│   → Use as-is, or min-max scale
│
├── Does it have a non-linear step-wise relationship?
│   → Bucketize into categories
│
└── Does it have extreme outliers?
    → Clip before any other processing
```

---

### 2.2 Categorical Features

Discrete values: country, device type, video category, language.

#### Technique 1: One-Hot Encoding

```python
# device_type ∈ {mobile, tablet, desktop, TV}
# mobile  → [1, 0, 0, 0]
# desktop → [0, 0, 1, 0]

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(categories.reshape(-1, 1))
```

**When to use**: Low cardinality (< 20 values). Simple, no training needed.
**Problem**: Breaks with high cardinality (100K categories → 100K-dim sparse vector).

#### Technique 2: Embedding Lookup (Neural Networks)

```python
import torch.nn as nn

# Map each category to a learned dense vector
category_emb = nn.Embedding(num_categories=200, embedding_dim=16)

# "gaming" → index 5 → learned vector [0.8, 0.1, -0.2, ...]
# Similar categories end up with similar embeddings after training
```

**When to use**: Neural network models, any cardinality. The standard approach.

Embedding dimension rule of thumb:

```
Cardinality    Embedding Dim    Example
< 10           4-8              device_type, age_bucket
10-100         8-16             country, video_category
100-10K        16-32            creator_id, city
10K-1M         32-64            video_id (small corpus)
1M+            64 + hashing     user_id, video_id (large corpus)
```

#### Technique 3: Label Encoding (Tree Models)

```python
# Assign integers: "gaming"=0, "cooking"=1, "music"=2
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
encoded = le.fit_transform(categories)

# LightGBM/CatBoost can handle this directly as categorical
# XGBoost needs one-hot or target encoding
```

#### Technique 4: Target Encoding (Tree Models)

Replace each category with the mean of the target variable for that category.

```python
# category → mean(label) for that category
# "gaming"  → 0.72  (72% click rate for gaming videos)
# "cooking" → 0.45  (45% click rate for cooking videos)

def target_encode(train_df, col, target_col, smoothing=10):
    """Target encoding with smoothing to prevent overfitting on rare categories."""
    global_mean = train_df[target_col].mean()
    stats = train_df.groupby(col)[target_col].agg(["mean", "count"])
    # Smoothed: blend category mean with global mean based on count
    stats["smoothed"] = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
    return train_df[col].map(stats["smoothed"])
```

**When to use**: Tree models with high-cardinality categories. Risk of target leakage — always compute on training fold only, apply to validation/test.

#### Technique 5: Hash Encoding (Very High Cardinality)

```python
# When you have 100M user_ids, you can't have a 100M-row embedding table
# Solution: hash into a fixed-size bucket

def hash_encode(value, num_buckets=1_000_000):
    return hash(value) % num_buckets

# "user_abc123" → hash → bucket 472831 → embedding[472831]
# Collisions happen but are tolerable at large bucket sizes
```

**When to use**: IDs with cardinality > 1M where a full embedding table is too expensive.

---

### 2.3 Time / Date Features

Timestamps, dates, day of week, hour of day.

#### Technique 1: Extract Components

```python
from datetime import datetime

dt = datetime(2024, 3, 15, 20, 30, 0)  # Friday, March 15, 2024, 8:30 PM

features = {
    "hour":        dt.hour,         # 20
    "day_of_week": dt.weekday(),    # 4 (Friday)
    "month":       dt.month,        # 3
    "is_weekend":  dt.weekday() >= 5,  # False
    "is_morning":  6 <= dt.hour < 12,  # False
    "is_evening":  18 <= dt.hour < 24, # True
}
```

Then treat each as categorical (embedding) or numeric.

#### Technique 2: Cyclical Encoding (Important!)

Hours and months are **cyclical** — hour 23 is close to hour 0, December is close to January. Standard numeric encoding doesn't capture this.

```python
import numpy as np

def cyclical_encode(value, max_value):
    """
    Encode a cyclical feature using sin/cos.
    hour=23 and hour=0 will have similar encodings.
    """
    sin_val = np.sin(2 * np.pi * value / max_value)
    cos_val = np.cos(2 * np.pi * value / max_value)
    return sin_val, cos_val

# Hour of day (0-23)
hour_sin, hour_cos = cyclical_encode(hour, 24)

# Day of week (0-6)
dow_sin, dow_cos = cyclical_encode(day_of_week, 7)

# Month (1-12)
month_sin, month_cos = cyclical_encode(month, 12)

# Day of year (1-365)
doy_sin, doy_cos = cyclical_encode(day_of_year, 365)

# Example:
# hour=0  → sin=0.00, cos=1.00
# hour=6  → sin=1.00, cos=0.00
# hour=12 → sin=0.00, cos=-1.00
# hour=23 → sin=-0.26, cos=0.97  ← close to hour=0!
```

**When to use**: Any feature that wraps around — hours, days of week, months, angles.
**Alternative**: Treat as categorical with embedding. Works well and is simpler. Cyclical encoding is more principled when using linear models.

#### Technique 3: Time Deltas (Relative Time)

Often more useful than absolute time.

```python
# "How long ago was this video uploaded?"
upload_age_days = (now - upload_time).total_seconds() / 86400

# "How long since the user's last session?"
session_gap_hours = (now - last_session_time).total_seconds() / 3600

# "How long since the user watched a video in this category?"
category_recency_days = (now - last_watched_category_time).total_seconds() / 86400

# All of these should be log-transformed (recency follows power law)
upload_age_feature = np.log1p(upload_age_days)
```

**When to use**: Freshness signals, recency features, session modeling. Time deltas are almost always more predictive than absolute timestamps.

---

### 2.4 Text Features

Video titles, descriptions, search queries, comments.

#### Technique 1: Pretrained Sentence Embeddings (Recommended)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

title = "How to Build a Transformer from Scratch"
title_emb = model.encode(title)  # 384-d dense vector

# Project to smaller dimension if needed
# 384-d → 128-d via a learned linear layer in the model
```

**When to use**: Always the first choice for text features in neural networks. Captures semantic meaning ("PyTorch tutorial" is close to "TensorFlow guide").

#### Technique 2: TF-IDF + Dimensionality Reduction

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Step 1: TF-IDF (sparse, high-dimensional)
tfidf = TfidfVectorizer(max_features=10000)
sparse_matrix = tfidf.fit_transform(titles)  # (N, 10000) sparse

# Step 2: Reduce dimension with SVD
svd = TruncatedSVD(n_components=128)
dense_emb = svd.fit_transform(sparse_matrix)  # (N, 128) dense
```

**When to use**: Baseline, simpler pipeline, tree models (which can't use pretrained transformers directly).

#### Technique 3: Bag of Words / Count Features

```python
# Simple but sometimes effective for tree models
text_features = {
    "title_length":      len(title.split()),
    "has_question_mark":  "?" in title,
    "has_exclamation":    "!" in title,
    "has_number":         any(c.isdigit() for c in title),
    "all_caps_words":     sum(1 for w in title.split() if w.isupper()),
    "title_char_count":   len(title),
}
```

**When to use**: Quick features for tree models. Captures surface patterns (clickbait often uses ALL CAPS and exclamation marks).

---

### 2.5 ID Features

User IDs, video IDs, creator IDs, session IDs.

```
Key principle: IDs themselves have NO inherent meaning.
  "user_abc123" tells the model nothing.
  But after training, the EMBEDDING for user_abc123 captures
  everything the model learned about this user's preferences.
```

#### Low Cardinality (< 1M): Direct Embedding

```python
user_emb = nn.Embedding(num_users, 64)      # Each user gets a 64-d vector
video_emb = nn.Embedding(num_videos, 64)     # Each video gets a 64-d vector
creator_emb = nn.Embedding(num_creators, 32) # Each creator gets a 32-d vector
```

#### High Cardinality (> 1M): Hash Embedding

```python
class HashEmbedding(nn.Module):
    """Multiple hash functions to reduce collision impact."""
    def __init__(self, num_buckets=1_000_000, embed_dim=64, num_hashes=2):
        super().__init__()
        self.num_hashes = num_hashes
        self.tables = nn.ModuleList([
            nn.Embedding(num_buckets, embed_dim)
            for _ in range(num_hashes)
        ])

    def forward(self, ids):
        embs = []
        for i, table in enumerate(self.tables):
            hashed = (ids * (i + 1) * 2654435761) % table.num_embeddings
            embs.append(table(hashed))
        return sum(embs) / self.num_hashes  # Average of multiple hash lookups
```

#### For Tree Models: Don't Use Raw IDs

Tree models can't learn embeddings. Instead, use **aggregated statistics** about the ID:

```python
# Instead of user_id directly, compute:
user_features = {
    "user_total_watches":       1250,
    "user_avg_watch_time":      340.5,
    "user_num_liked":           89,
    "user_top_category":        "gaming",     # categorical
    "user_days_since_signup":   365,
    "user_num_sessions_7d":     12,
}
```

---

### 2.6 Sequence / List Features

Watch history, search history, click history — ordered lists of items.

#### Technique 1: Average Pooling

```python
# Average of the last N item embeddings
def avg_pool_history(item_ids, embedding_table, max_len=50):
    recent = item_ids[-max_len:]
    embs = embedding_table(torch.tensor(recent))  # (L, D)
    return embs.mean(dim=0)  # (D,)

# Pros: Simple, O(1) at serving if pre-aggregated
# Cons: Loses order — [A, B, C] same as [C, B, A]
```

#### Technique 2: Weighted Average (Time Decay)

```python
# Recent items weighted more heavily
def weighted_avg_history(items_with_time, embedding_table, half_life_days=7):
    now = time.time()
    weighted_sum = np.zeros(embed_dim)
    total_weight = 0
    for item_id, timestamp in items_with_time[-50:]:
        age_days = (now - timestamp) / 86400
        weight = 2 ** (-age_days / half_life_days)
        weighted_sum += embedding_table[item_id] * weight
        total_weight += weight
    return weighted_sum / (total_weight + 1e-10)
```

#### Technique 3: Aggregated Statistics

```python
# For tree models (can't use embeddings directly)
history_features = {
    "num_videos_watched_7d":       45,
    "num_unique_categories_7d":    6,
    "avg_watch_time_7d":           320.0,
    "max_watch_time_7d":           1800.0,
    "pct_completed_7d":            0.42,
    "most_watched_category":       "gaming",
    "num_liked_7d":                8,
    "num_disliked_7d":             1,
    "sessions_per_day_7d":         2.3,
}
```

#### Technique 4: Transformer Encoding (Best Quality)

```python
# Self-attention over history — captures sequential patterns
# See 08_two_tower_training_deep_dive.md for full code
class HistoryEncoder(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, history_embs):
        return self.transformer(history_embs)[:, -1, :]  # Last position as summary
```

---

### 2.7 Boolean / Binary Features

True/false flags.

```python
# Simple: use 0/1 directly
features = {
    "user_is_premium":       1,
    "video_has_subtitles":   0,
    "is_weekend":            1,
    "user_is_new":           1,   # signed up < 7 days ago
    "video_is_live":         0,
    "user_subscribed_to_creator": 1,
}

# No transformation needed — they're already numeric.
# Can also be used as-is in tree models.
```

---

### 2.8 Geographic / Location Features

Country, city, coordinates, IP addresses.

```python
# Country: categorical embedding (top 50 + "other")
country_emb = nn.Embedding(51, 16)

# City: too many → hash embedding or region grouping
# Group into regions: "San Francisco" → "US_West"

# Lat/Lon: use directly as 2 numeric features
# Or compute derived features:
geo_features = {
    "latitude":           37.77,
    "longitude":          -122.42,
    "timezone_offset":    -8,
    "is_urban":           True,
    "distance_to_creator_km":  2500.0,  # geographic relevance
}
```

---

### 2.9 Currency / Money Features

Prices, revenue, budgets.

```python
money_features = {
    # Log transform (prices follow power law)
    "price_log":             np.log1p(price),

    # Relative to category average
    "price_vs_category_avg": price / category_avg_price,

    # Bucketize
    "price_bucket":          "$$" if 10 < price < 30 else "$$$$",

    # Currency normalization (if multi-currency)
    "price_usd":             price * exchange_rate,
}
```

---

### 2.10 Phone Numbers / Social IDs / Emails

These are usually **identifiers**, not features. Don't feed raw values to a model.

```python
# DON'T: use "555-0123" as a feature — meaningless digits
# DO: extract derived signals

phone_features = {
    "has_verified_phone":    True,   # binary
    "phone_country_code":    "+1",   # categorical → country signal
}

social_features = {
    "has_linked_social":     True,   # binary
    "num_social_connections": 342,   # numeric → log transform
    "social_platform":       "twitter",  # categorical
}

email_features = {
    "email_domain":          "gmail.com",  # categorical (top 20 + other)
    "has_verified_email":    True,
}
```

---

## 3. Cross Features (Feature Interactions)

Some of the most powerful features come from **combining** two or more features.

```python
# Cross features capture interactions the model might miss

cross_features = {
    # User-Item crosses (MOST IMPORTANT for ranking)
    "user_watched_this_creator_before":   True,
    "user_category_watch_fraction":       0.35,   # what % of user's watches are this category
    "user_avg_watch_time_in_category":    450.0,  # how long user watches this category
    "two_tower_similarity_score":         0.82,   # dot(user_emb, item_emb)

    # Time-Context crosses
    "is_commute_time_and_mobile":         True,   # 7-9am + mobile → short videos
    "is_evening_and_desktop":             True,   # 8-11pm + desktop → long videos

    # User-Context crosses
    "user_is_new_and_cold_start":         True,   # new user + no history
}
```

**For neural networks**: The model can learn interactions from raw features, but explicitly providing common crosses gives it a head start.

**For tree models**: Trees naturally find interactions (each split path is a cross), but providing them explicitly helps with shallow trees.

---

## 4. Feature Freshness Tiers

Not all features can be computed in real-time. Design your feature pipeline around freshness tiers:

```
Tier          Refresh Rate        Examples                            Storage
──────────    ──────────────      ──────────────────────────          ────────
Real-time     Per request         device, hour, geo, query           Compute on-the-fly
Streaming     Seconds-minutes     last 3 videos watched,             Redis / Kafka
                                  current session length
Near-RT       Minutes-hours       watch_history_emb,                 Redis / feature store
                                  category_affinity
Daily batch   Once per day        user_avg_watch_time_7d,            Hive → Redis
                                  video engagement stats
Static        Rarely changes      user country, video title_emb,     Embedded in model
                                  creator info                        or feature store
```

---

## 5. Complete Cheat Sheet

| Raw Data Type | Technique | Output | Model Type |
|---|---|---|---|
| **Count/Amount** (views, time) | log1p → z-score | 1 float | Both |
| **Percentage/Ratio** (completion rate) | Use directly or min-max | 1 float | Both |
| **Category** (< 20 values) | One-hot | K binary | Both |
| **Category** (20-10K) | Embedding (NN) / Label encode (tree) | 4-32 floats | NN / Tree |
| **Category** (> 10K) | Hash embedding (NN) / Target encode (tree) | 32-64 floats | NN / Tree |
| **ID** (user, video) | Embedding or hash embedding | 32-64 floats | NN |
| **ID** (for tree models) | Aggregated stats about the ID | N floats | Tree |
| **Hour/Day/Month** | Cyclical sin/cos or embedding | 2 floats or 4-8 | Both |
| **Timestamp** | Time delta (age/recency) → log1p | 1 float | Both |
| **Text** (title) | Sentence embedding (NN) / TF-IDF+SVD (tree) | 128-384 floats | Both |
| **Sequence** (history) | Avg pool / weighted avg / transformer | 128 floats | NN |
| **Sequence** (for tree) | Aggregated stats (count, avg, max) | N floats | Tree |
| **Boolean** | 0/1 directly | 1 int | Both |
| **Geo** (country) | Embedding (categorical) | 8-16 floats | Both |
| **Geo** (lat/lon) | Use directly or region grouping | 2 floats | Both |
| **Money** | log1p, relative to avg, bucketize | 1-3 floats | Both |
| **Phone/Email/Social** | Extract derived signals only | varies | Both |

---

## 6. Video Recommendation Example: Full Feature Vector

Here is the complete feature engineering for our video recommendation ranking model, showing every raw field and how it becomes model input.

### 6.1 User Features

```
Raw Data                          Technique                  Output        Dim
──────────────────────────────    ─────────────────────      ──────        ───
user_id = "u_382910"              Embedding(10M, 64)         dense vec     64
age = 28                          Bucketize [18-24,25-34,..] embedding     8
                                  → "25-34" → Embedding(6,8)
country = "US"                    Embedding(51, 16)          dense vec     16
language = "en"                   Embedding(30, 8)           dense vec     8
device = "mobile"                 One-hot [mob,tab,desk,TV]  binary vec    4
avg_watch_time_7d = 1250 sec      log1p → z-score            float         1
num_videos_watched_7d = 85        log1p → z-score            float         1
watch_history = [v1..v50]         avg(emb(v1)..emb(v50))     dense vec     128
category_affinity                 Normalized distribution    float vec     20
  {gaming:0.3, ml:0.25, ...}     over 20 categories
hour_of_day = 20                  Embedding(24, 8) or        dense vec     8
                                  cyclical sin/cos                         (or 2)
day_of_week = 5 (Friday)          Embedding(7, 4)            dense vec     4
                                                             ─────────────────
                                                             Total:        ~262
```

### 6.2 Video Features

```
Raw Data                          Technique                  Output        Dim
──────────────────────────────    ─────────────────────      ──────        ───
video_id = "v_19283"              Embedding(5M, 64)          dense vec     64
creator_id = "c_4521"             Embedding(500K, 32)        dense vec     32
category = "ml_tech"              Embedding(200, 16)         dense vec     16
duration = 480 sec                Bucketize [<1m,1-5m,...]   embedding     8
                                  → "5-15m" → Embedding(5,8)
upload_age = 12 days              log1p → z-score            float         1
title = "PyTorch Tips"            SentenceTransformer        dense vec     128
                                  → project to 128-d
total_views = 1,250,000           log1p → z-score            float         1
avg_completion_rate = 0.72        Use directly               float         1
like_ratio = 0.89                 Use directly               float         1
                                                             ─────────────────
                                                             Total:        ~252
```

### 6.3 Cross Features (User × Video)

```
Raw Data                          Technique                  Output        Dim
──────────────────────────────    ─────────────────────      ──────        ───
user_watched_this_creator         Binary lookup              0 or 1        1
user_pct_category_watches         Fraction: user's watches   float         1
                                  in this video's category
two_tower_similarity              dot(user_emb, video_emb)   float         1
                                  from retrieval model
time_since_last_category_watch    log1p(hours)               float         1
user_avg_watch_time_for_category  log1p → z-score            float         1
retrieval_source                  One-hot [two_tower,        binary vec    4
                                  content, popularity, sub]
position_in_retrieval_results     log1p → z-score            float         1
                                                             ─────────────────
                                                             Total:        ~10
```

### 6.4 Final Input to Ranking Model

```
user_features (262) + video_features (252) + cross_features (10) = 524-dim input vector

This 524-dim vector is the input to the MMoE ranking model for ONE (user, video) pair.

For 300 candidates: input shape = (300, 524)
Output: 5 task predictions per candidate → sort by weighted score.
```

### 6.5 Feature Importance (What Matters Most)

From industry experience (YouTube, Netflix), ranked by impact:

```
Tier 1 — Highest impact (add these first):
  1. two_tower_similarity_score     ← collaborative filtering signal
  2. user_watched_this_creator      ← direct user-item history
  3. user_pct_category_watches      ← user-category affinity
  4. avg_completion_rate             ← video quality proxy
  5. watch_history_emb               ← user behavioral profile

Tier 2 — Strong signal:
  6. like_ratio                      ← video quality
  7. total_views_log                 ← popularity signal
  8. title_emb                       ← content understanding
  9. hour_of_day + device            ← context
  10. upload_age                     ← freshness

Tier 3 — Incremental gains:
  11. user demographics (age, country)
  12. video duration bucket
  13. creator features
  14. session context (nth video in session)
```

---

## 7. Common Mistakes

| Mistake | Why It's Bad | Fix |
|---|---|---|
| Using raw IDs as numeric features | "user_382910" treated as a large number — meaningless | Embedding lookup |
| Not log-transforming skewed features | Outliers dominate gradients, slow convergence | log1p before normalizing |
| Using absolute timestamps | "1700000000" is meaningless to the model | Extract components or compute time deltas |
| One-hot for high-cardinality | 100K-dim sparse vector, memory explosion | Embedding or hash encoding |
| Training-time feature != serving-time feature | Model degrades silently in production | Log serving features, reuse for training |
| Leaking future information | Using tomorrow's data to predict today | Strict temporal splits, check feature timestamps |
| Not handling missing values | NaN propagates through computations | Impute (median/0), add "is_missing" binary flag |

---

## 8. Interview Talking Points

### "Walk me through your feature engineering process."

> I start by categorizing each raw field: is it numeric, categorical, text, temporal, or a sequence? Then I apply standard techniques — log transform for skewed numerics, embeddings for categoricals, sentence encoders for text, and cyclical encoding for time features. The highest-leverage features are usually **cross features** between user and item — things like "has this user watched this creator before" or "what fraction of this user's watches are in this category." These directly capture the user-item relationship, not just properties of each independently.

### "What's the most important feature in a ranking model?"

> User-item interaction history features. "Has this user engaged with this creator/category/topic before?" dominates all other signals. After that, the retrieval model's similarity score (dot product from Two-Tower) is extremely powerful because it compresses all the collaborative filtering signal into one number. Video quality metrics (completion rate, like ratio) come third.

### "How do you handle a feature with 100M unique values?"

> Hash embedding. Map each value to a fixed-size bucket table (e.g., 1M buckets) using hash functions. Collisions are tolerable — at 1M buckets with 100M values, each bucket holds ~100 values on average, but the embedding still captures useful group-level patterns. Using multiple hash functions and averaging reduces collision impact.

### "Why log transform?"

> Most real-world counts and amounts follow power-law distributions — a few items have millions of views while most have hundreds. Without log transform, the gradient is dominated by extreme values, the model has trouble distinguishing 100 from 200 views because it's too busy fitting the 10M-view outlier. Log compresses the range so differences at all scales contribute equally to learning.
