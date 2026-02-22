# 04 — Data Preparation

## Data Sources

### 1. Graph Data (most important)
```
Edge list: (user_A, user_B, connected_at, connection_type)
- connection_type: direct connection, follow, colleague, classmate
- Scale: ~500B edges for 1B users × 1000 avg connections
```

### 2. User Profile Data
```
- user_id, industry, job_title, company, location
- education (school, graduation_year, degree)
- skills (list of free-text or standardized tags)
- profile_completeness_score
- account_age, is_verified
```

### 3. Behavioral / Interaction Data
```
- profile_views:    (viewer_id, viewed_id, timestamp, duration_seconds)
- connection_events:(requester_id, target_id, action, timestamp)
  action ∈ {sent, accepted, rejected, withdrawn, ignored}
- content_engagement: (user_id, content_author_id, action, timestamp)
  action ∈ {like, comment, share, click}
- search_clicks:    (searcher_id, clicked_profile_id, timestamp)
- message_events:   (sender_id, recipient_id, timestamp)
```

### 4. Negative Signals
```
- dismiss_events:   (user_id, dismissed_candidate_id, timestamp)
  — explicit "Not Interested" clicks on PYMK widget
- block_events:     (blocker_id, blocked_id, timestamp)
```

---

## Label Construction

Label generation is one of the most critical (and often underestimated) steps. Poor labels = poor model regardless of architecture.

### Positive Labels

| Event | Strength | Notes |
|---|---|---|
| Connection accepted | Strong positive | Clean, unambiguous intent |
| Profile viewed + connection sent | Strong positive | Two-signal confirmation |
| Profile viewed (≥30s) | Weak positive | Interest without action |
| Content engaged (multiple times) | Weak positive | Interest in content, not necessarily person |

### Negative Labels

This is harder. The options:

**Option A: Random negatives**
- Randomly sample users not connected to the query user
- Easy to generate, but most are trivially easy (unrelated people)
- Model learns to distinguish "clearly unrelated" vs. "clearly related" — not useful at serving time

**Option B: Impressed-but-not-converted (recommended)**
- Negative = user was shown a recommendation (impression) but did not connect over N days
- Much harder negatives — the model was already confident enough to show them
- Reduces false negative rate (the user might have connected later)
- Use a 30-day window: if no connection formed after 30 days of impression, label as negative

**Option C: Explicit dismissals**
- User clicked "Not Interested"
- Strongest negative signal
- But very sparse (most users never click dismiss, they just scroll past)

**Recommended label strategy:**
```
Positive:  accepted connection (or sent + accepted within 7 days)
Negative:  impression with no action after 14 days
           + explicit dismissals (upweighted 3×)

Exclude from training:
  - pending requests (ambiguous)
  - recently shown impressions (< 14 days, outcome unknown)
  - already-connected pairs (obvious)
```

---

## Data Sampling Strategies

At 1B users, you cannot train on all data. Sampling strategy matters enormously.

### 1. User Sampling for Training
- Sample a representative subset of users per day
- Stratify by: user tenure, activity level, connection count
- Ensure cold-start users (0–5 connections) are oversampled — they're hardest to model

### 2. Negative Sampling Ratio
- Typical ratio: 1 positive : 4–10 negatives
- Too few negatives: model is overconfident
- Too many negatives: training signal diluted
- For GNN training specifically, use **hard negatives** in later training epochs (see Model Development)

### 3. Temporal Split for Evaluation
```
Training data:   Jan 1  – Dec 31 (full year)
Validation data: Jan 1  – Jan 31 of next year  (1 month out)
Test data:       Feb 1  – Feb 28                (2 months out)
```
**Never use random train/test splits for recommendation systems.** This causes data leakage — the model sees future connections during training.

### 4. Graph Snapshot Sampling
For GNN training, you cannot load the full 500B-edge graph into memory. Strategies:
- **Mini-batch subgraph sampling:** For each training example (user pair), sample a computation subgraph (the k-hop neighborhood of both users)
- **Historical snapshot:** Train on a daily or weekly graph snapshot; deploy with the current graph
- **Incremental training:** Fine-tune on the delta (new edges) rather than retraining from scratch

---

## Data Storage Architecture

```
Graph Data
  └── Graph DB (e.g., Neo4j, Amazon Neptune, custom adjacency list)
      Purpose: neighbor lookups, graph traversal
      Access pattern: "give me all neighbors of user X"
      Sharded by: user_id hash

Profile Data
  └── Document DB (e.g., MongoDB, DynamoDB)
      Purpose: feature retrieval at serving time
      Access pattern: "give me profile features for user X"

Interaction Logs
  └── Data Warehouse (e.g., Hive, BigQuery, Spark tables)
      Purpose: offline label generation, feature engineering
      Access pattern: batch scans, aggregations

Training Dataset
  └── Columnar files (Parquet on S3/GCS)
      Schema: (query_user_id, candidate_user_id, label, features...)
      Partitioned by: date
```

---

## Data Quality Issues

| Issue | Impact | Mitigation |
|---|---|---|
| Bot accounts | Pollutes graph structure | Filter accounts flagged by trust & safety |
| Fake connections | Adds spurious edges | Weight connections by engagement, not just existence |
| Profile staleness | Features don't reflect current user | Use recency-weighted features; track last_updated |
| Selection bias in labels | Impressions are already filtered by current model | Re-weight samples by inverse propensity score |
| Graph sparsity for new users | Cold start | Profile-only features as fallback |

---

## Interview Checkpoint

**Q: How do you handle selection bias in your training labels?**

The labels (impressions that did/didn't convert) come from the current production system, which already filters candidates. This means the training set is biased toward what the old model thought was good. New, potentially better candidates never appear as negative examples because they were never shown.

The fix is **inverse propensity scoring (IPS)**: weight each training example by the inverse of the probability that it was selected by the current system. This de-biases the training distribution. LinkedIn has published work on this for their feed ranking.

**Q: Why is temporal split important for recommendation systems?**

If you randomly split connections into train/test, you might have the following situation: the connection between Alice and Bob is in the test set, but Alice's profile view of Bob is in the training set. The model sees a signal that "leaks" the test label. Temporal splits ensure all training data strictly precedes all evaluation data, mimicking real deployment conditions.
