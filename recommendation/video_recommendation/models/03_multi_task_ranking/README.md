# Multi-Task Ranking Model (MMoE / PLE)

## 1. Role in the Pipeline

The Multi-Task Ranking Model is the **core of Stage 2 (Ranking)**. It receives ~300 candidates from all retrieval sources and scores each one by predicting **multiple engagement signals simultaneously**:

- **P(click)** — Will the user click on this video?
- **E[watch_time]** — How long will the user watch?
- **P(like)** — Will the user like this video?
- **P(finish)** — Will the user watch to completion?

The final ranking score is a weighted combination of these predictions.

**Why multi-task?**
- Different engagement signals are **correlated but distinct** — a user may click (curious) but not finish (disappointed)
- Shared representation learning across tasks improves each individual prediction
- Enables tunable business tradeoffs: adjust weights without retraining
- **Used at: YouTube, TikTok (Monolith), Instagram Explore, Kuaishou**

```
          Input Features (user + video + context)
                         │
              ┌──────────┼──────────┐
              │     Shared Backbone   │
              │   (Expert Networks)   │
              └──────────┼──────────┘
                         │
         ┌───────┬───────┼───────┬────────┐
         │       │       │       │        │
      Gate_1  Gate_2  Gate_3  Gate_4   Gate_5
         │       │       │       │        │
      Tower_1 Tower_2 Tower_3 Tower_4  Tower_5
         │       │       │       │        │
     P(click) E[watch] P(like) P(finish) P(dislike)
```

---

## 2. Data Requirements

### Training Data

Each training example is a **(user, video, context) → engagement outcome** record generated from impression logs.

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | string | User identifier |
| `video_id` | string | Video identifier |
| `impression_id` | string | Unique impression event |
| `timestamp` | int64 | When the impression occurred |
| `position` | int | Position in the feed (for position bias correction) |
| `clicked` | int | 1 if user clicked, 0 otherwise |
| `watch_time_sec` | float | Seconds watched (0 if not clicked) |
| `video_duration_sec` | float | Total video duration |
| `liked` | int | 1 if user liked, 0 otherwise |
| `finished` | int | 1 if user watched > 80% of video, 0 otherwise |
| `disliked` | int | 1 if user disliked, 0 otherwise |

### Label Construction

```python
labels = {
    "click":      int(row["clicked"] == 1),                    # Binary
    "watch_time": log(1 + row["watch_time_sec"]),              # Regression (log-transformed)
    "like":       int(row["liked"] == 1),                       # Binary
    "finish":     int(row["watch_time_sec"] / row["video_duration_sec"] > 0.8),  # Binary
    "dislike":    int(row["disliked"] == 1),                    # Binary
}
```

**Key: Watch time is log-transformed** because raw watch time is heavily skewed (most videos get <10s, some get >30min). Log-transform makes the regression target more Gaussian.

### Label Hierarchy & Selection Bias

```
Impression (shown to user)
    └── Click? (selection bias: post-click labels only exist if clicked)
         ├── Watch time
         ├── Like?
         ├── Finish?
         └── Dislike?
```

**Critical interview point**: Post-click labels (watch_time, like, finish) are **conditionally observed**. You only see watch_time if the user clicked. Naive training on all impressions treats non-clicked as watch_time=0, which is incorrect.

**Solutions**:
1. Train click model on all impressions; train post-click models only on clicked impressions
2. Use **inverse propensity weighting** to correct for selection bias
3. Multi-task learning naturally handles this if loss masking is done correctly

### Scale

- Training: ~1B impressions/day → sample to ~100M–500M for training
- Refresh: Retrain daily or every few hours on fresh data

### Example Raw Data

```json
{
  "impression_id": "imp_88291002",
  "user_id": "u_382910",
  "video_id": "v_19283",
  "timestamp": 1700000000,
  "position": 3,
  "clicked": 1,
  "watch_time_sec": 187.5,
  "video_duration_sec": 245.0,
  "liked": 1,
  "finished": 0,
  "disliked": 0
}
```

---

## 3. Feature Engineering

### Feature Groups

#### User Profile Features

| Feature | Type | Processing |
|---------|------|------------|
| `user_id_emb` | Embedding | Lookup (64-d) |
| `age_bucket` | Categorical | Embed [0-17, 18-24, 25-34, 35-44, 45-54, 55+] (8-d) |
| `country` | Categorical | Embed top-50 (16-d) |
| `language` | Categorical | Embed (8-d) |
| `device_type` | Categorical | One-hot [mobile, tablet, desktop, TV] (4-d) |
| `account_age_days` | Continuous | Log-transform, normalize (1-d) |

#### User Behavioral Features (Real-Time)

| Feature | Type | Processing |
|---------|------|------------|
| `avg_session_watch_time` | Continuous | Rolling 7-day average (1-d) |
| `avg_video_completion_rate` | Continuous | Rolling 7-day (1-d) |
| `videos_watched_last_24h` | Continuous | Count, log-transform (1-d) |
| `category_watch_dist` | Dense vector | Normalized distribution over categories (20-d) |
| `last_n_video_embs` | Sequence | Average of last 20 watched video CF embeddings (128-d) |
| `time_since_last_session` | Continuous | Log-transform minutes (1-d) |
| `click_through_rate_7d` | Continuous | Personal CTR (1-d) |

#### Video Features

| Feature | Type | Processing |
|---------|------|------------|
| `video_id_emb` | Embedding | Lookup (64-d) |
| `creator_id_emb` | Embedding | Lookup (32-d) |
| `category` | Categorical | Embed (16-d) |
| `video_duration_sec` | Continuous | Log-transform (1-d) |
| `upload_age_hours` | Continuous | Log-transform (1-d) |
| `title_emb` | Dense | Pretrained text embedding (128-d) |
| `thumbnail_emb` | Dense | Pretrained visual embedding (128-d) |

#### Video Engagement Statistics

| Feature | Type | Processing |
|---------|------|------------|
| `video_ctr` | Continuous | Historical CTR (1-d) |
| `video_avg_watch_pct` | Continuous | Average completion rate (1-d) |
| `video_like_rate` | Continuous | likes / impressions (1-d) |
| `video_views_log` | Continuous | Log(views + 1) (1-d) |
| `video_dislike_rate` | Continuous | dislikes / impressions (1-d) |

#### Context Features

| Feature | Type | Processing |
|---------|------|------------|
| `hour_of_day` | Categorical | Embed (8-d) |
| `day_of_week` | Categorical | Embed (4-d) |
| `is_weekend` | Binary | 0/1 (1-d) |
| `position_in_feed` | Continuous | Normalized (1-d) |

#### Cross Features (Interaction)

| Feature | Type | Processing |
|---------|------|------------|
| `user_x_category_affinity` | Continuous | User's watch-time share in this video's category (1-d) |
| `user_x_creator_history` | Continuous | # videos user watched from this creator (1-d) |
| `user_x_duration_preference` | Continuous | Abs diff between user's avg watch duration and video duration (1-d) |
| `user_cf_emb · video_cf_emb` | Continuous | Dot product of CF embeddings (1-d) |

### Total Feature Dimension

~650–700 dimensional input after concatenation.

---

## 4. Model Architecture Options

### Option A: Multi-gate Mixture of Experts (MMoE) — Google 2018

**The industry standard for multi-task ranking.**

```
Input features (d=700)
         │
    ┌────┼────┬────┐
    │    │    │    │
  Expert Expert Expert Expert    ← Shared expert networks (each: 2-layer MLP)
  (512)  (512) (512) (512)       ← N=4-8 experts typical
    │    │    │    │
    └────┼────┴────┘
         │
   ┌─────┼─────┬─────┬──────┐
   │     │     │     │      │
 Gate_1 Gate_2 Gate_3 Gate_4 Gate_5  ← Per-task gating: softmax over experts
   │     │     │     │      │
Tower_1 Tower_2 Tower_3 Tower_4 Tower_5  ← Task-specific towers
   │     │     │     │      │
P(click) E[wt] P(like) P(fin) P(dis)   ← Task outputs
```

**Key insight**: Each gate learns to weight experts differently per task, allowing task-specific expert utilization without full parameter separation.

### Option B: Progressive Layered Extraction (PLE) — Tencent 2020

Extends MMoE with **task-specific experts** alongside shared experts:

```
Input features
         │
    ┌────┼────────────────┐
    │    │                │
 Shared  Task-1-specific  Task-2-specific  ← Both shared AND task-specific experts
 Experts   Experts          Experts
    │    │                │
    └────┼────────────────┘
         │ (per-task gating over shared + own task experts)
    ┌────┼────┐
  Gate_1    Gate_2
    │         │
  Tower_1   Tower_2
    │         │
  Task_1    Task_2
```

Can be **stacked into multiple extraction layers** for progressive refinement.

### Option C: Shared-Bottom with Task Towers (Simple Baseline)

```
Input features
      │
  Shared MLP (3 layers)
      │
  ┌───┼───┬───┬───┐
  │   │   │   │   │
 T1  T2  T3  T4  T5  ← Task-specific tower heads
```

**Simpler but suffers from negative transfer** — tasks can hurt each other since all share the same representation.

### Architecture Comparison

| Aspect | Shared-Bottom | MMoE | PLE |
|--------|--------------|------|-----|
| Task interference | High | Low | Lowest |
| Parameter count | Lowest | Medium | Highest |
| Implementation complexity | Simple | Medium | Complex |
| Industry adoption | Baseline | **Most common** | Growing |
| Best for | < 3 similar tasks | 3-8 tasks | 5+ diverse tasks |

**Recommendation**: Start with **MMoE**. Move to PLE if you observe task interference (one task's metric degrades when adding a new task).

---

## 5. High-Level Training Code (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """Single expert network."""

    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class TaskTower(nn.Module):
    """Task-specific tower producing a single output."""

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 output_type: str = "binary"):
        super().__init__()
        self.tower = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.output_type = output_type

    def forward(self, x):
        logit = self.tower(x)
        if self.output_type == "binary":
            return torch.sigmoid(logit)
        else:  # regression
            return logit  # Raw output for regression


class MMoERankingModel(nn.Module):
    """
    Multi-gate Mixture of Experts ranking model.

    Predicts: P(click), E[watch_time], P(like), P(finish), P(dislike)
    """

    def __init__(self, input_dim: int, num_experts: int = 6,
                 expert_dim: int = 512, num_tasks: int = 5):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # Shared expert networks
        self.experts = nn.ModuleList([
            Expert(input_dim, expert_dim) for _ in range(num_experts)
        ])

        # Per-task gating networks
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, num_experts),
                nn.Softmax(dim=-1),
            )
            for _ in range(num_tasks)
        ])

        # Task-specific towers
        self.task_towers = nn.ModuleList([
            TaskTower(expert_dim, output_type="binary"),   # P(click)
            TaskTower(expert_dim, output_type="regression"),# E[watch_time]
            TaskTower(expert_dim, output_type="binary"),   # P(like)
            TaskTower(expert_dim, output_type="binary"),   # P(finish)
            TaskTower(expert_dim, output_type="binary"),   # P(dislike)
        ])

        self.task_names = ["click", "watch_time", "like", "finish", "dislike"]

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            features: (batch_size, input_dim) concatenated feature vector

        Returns:
            dict mapping task_name → prediction tensor (batch_size, 1)
        """
        # Compute all expert outputs
        expert_outputs = [expert(features) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (B, num_experts, D)

        # Per-task gating and tower
        predictions = {}
        for i, (gate, tower, name) in enumerate(
            zip(self.gates, self.task_towers, self.task_names)
        ):
            # Gate: soft selection of experts
            gate_weights = gate(features)  # (B, num_experts)
            gate_weights = gate_weights.unsqueeze(-1)  # (B, num_experts, 1)

            # Weighted sum of expert outputs
            gated_input = (expert_outputs * gate_weights).sum(dim=1)  # (B, D)

            # Task tower prediction
            predictions[name] = tower(gated_input)

        return predictions


# ─────────── Loss Functions ───────────

class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss with per-task weighting.

    Handles:
    - Binary tasks (click, like, finish, dislike): BCE loss
    - Regression tasks (watch_time): Huber loss
    - Selection bias: Only compute post-click losses on clicked samples
    """

    def __init__(self, task_weights: dict = None):
        super().__init__()
        self.task_weights = task_weights or {
            "click": 1.0,
            "watch_time": 1.0,
            "like": 1.0,
            "finish": 0.5,
            "dislike": 0.5,
        }
        self.bce = nn.BCELoss(reduction="none")
        self.huber = nn.SmoothL1Loss(reduction="none")

    def forward(self, predictions: dict, labels: dict,
                clicked_mask: torch.Tensor) -> dict:
        """
        Args:
            predictions: model outputs per task
            labels: ground truth per task
            clicked_mask: (B,) boolean, True if user clicked
        """
        losses = {}

        # Click loss: computed on ALL impressions
        click_loss = self.bce(predictions["click"], labels["click"])
        losses["click"] = click_loss.mean()

        # Post-click losses: ONLY on clicked impressions (selection bias handling)
        if clicked_mask.any():
            clicked = clicked_mask.bool()

            wt_loss = self.huber(
                predictions["watch_time"][clicked],
                labels["watch_time"][clicked]
            )
            losses["watch_time"] = wt_loss.mean()

            like_loss = self.bce(
                predictions["like"][clicked],
                labels["like"][clicked]
            )
            losses["like"] = like_loss.mean()

            finish_loss = self.bce(
                predictions["finish"][clicked],
                labels["finish"][clicked]
            )
            losses["finish"] = finish_loss.mean()

            dislike_loss = self.bce(
                predictions["dislike"][clicked],
                labels["dislike"][clicked]
            )
            losses["dislike"] = dislike_loss.mean()

        # Weighted total loss
        total = sum(
            self.task_weights.get(name, 1.0) * loss
            for name, loss in losses.items()
        )
        losses["total"] = total
        return losses


# ─────────── Training Loop ───────────

def train_ranking_model(model, train_loader, optimizer, device, epochs=5):
    """Training loop for the multi-task ranking model."""
    criterion = MultiTaskLoss()
    model.train()

    for epoch in range(epochs):
        epoch_losses = {k: 0.0 for k in ["total", "click", "watch_time", "like", "finish", "dislike"]}

        for batch in train_loader:
            features = batch["features"].to(device)
            labels = {k: batch[f"label_{k}"].to(device) for k in
                      ["click", "watch_time", "like", "finish", "dislike"]}
            clicked_mask = batch["label_click"].to(device).squeeze()

            predictions = model(features)
            losses = criterion(predictions, labels, clicked_mask)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k, v in losses.items():
                epoch_losses[k] += v.item()

        n = len(train_loader)
        print(f"Epoch {epoch+1}: " + " | ".join(
            f"{k}={v/n:.4f}" for k, v in epoch_losses.items()
        ))


# ─────────── Scoring at Serving Time ───────────

def compute_ranking_score(predictions: dict, weights: dict = None) -> float:
    """
    Combine multi-task predictions into a single ranking score.

    Weights are tuned via offline experiments + online A/B tests.
    """
    w = weights or {
        "click": 0.1,
        "watch_time": 1.0,  # Primary signal
        "like": 0.5,
        "finish": 0.3,
        "dislike": -2.0,    # Penalty
    }

    score = sum(w[task] * predictions[task] for task in w)
    return score
```

---

## 6. Model Input / Output Examples

### Training Input (Single Impression)

```python
features = {
    # User profile (one-hot / embedding indices, will be embedded)
    "user_id": "u_382910",
    "age_bucket": 2,        # index for "25-34"
    "country": 5,           # index for "US"
    "device_type": [1,0,0,0], # mobile

    # User behavioral (real-valued)
    "avg_session_watch_time": 4.2,    # log-transformed
    "videos_watched_last_24h": 3.4,   # log-transformed
    "category_watch_dist": [0.30, 0.25, 0.20, ...],  # 20-d
    "last_n_video_embs": [0.02, -0.11, ...],  # 128-d average

    # Video features
    "video_id": "v_19283",
    "creator_id": "c_4291",
    "category": 3,          # index for "education"
    "video_duration_sec": 5.5,  # log-transformed
    "title_emb": [0.03, -0.08, ...],  # 128-d

    # Video stats
    "video_ctr": 0.042,
    "video_avg_watch_pct": 0.68,

    # Context
    "hour_of_day": 20,
    "day_of_week": 5,
    "position_in_feed": 0.06,  # normalized

    # Cross features
    "user_x_category_affinity": 0.30,
    "user_x_creator_history": 1.6,  # log(1 + 4 videos)
}

labels = {
    "click": 1,
    "watch_time": 5.24,      # log(1 + 187.5)
    "like": 1,
    "finish": 0,             # 187.5 / 245.0 = 0.765 < 0.8
    "dislike": 0,
}
```

### Inference Output

For a single (user, candidate_video) pair:

```python
predictions = {
    "click": 0.72,        # 72% probability of clicking
    "watch_time": 4.83,   # log-space → exp(4.83)-1 ≈ 124 seconds expected
    "like": 0.18,         # 18% probability of liking
    "finish": 0.45,       # 45% probability of finishing
    "dislike": 0.02,      # 2% probability of disliking
}

# Combined score with default weights:
# 0.1*0.72 + 1.0*4.83 + 0.5*0.18 + 0.3*0.45 + (-2.0)*0.02
# = 0.072 + 4.83 + 0.09 + 0.135 - 0.04
# = 5.087
ranking_score = 5.087
```

### Scoring 300 Candidates

```python
# Input: 300 candidates from retrieval stage
candidates = [
    {"video_id": "v_19283", "retrieval_source": "two_tower", "retrieval_score": 0.92},
    {"video_id": "v_88421", "retrieval_source": "content_based", "retrieval_score": 0.89},
    # ... 298 more
]

# Output: All 300 scored and sorted
ranked_candidates = [
    {"video_id": "v_48291", "ranking_score": 7.32, "p_click": 0.85, "e_watch": 5.9, ...},
    {"video_id": "v_19283", "ranking_score": 5.09, "p_click": 0.72, "e_watch": 4.8, ...},
    {"video_id": "v_30291", "ranking_score": 4.88, "p_click": 0.68, "e_watch": 4.5, ...},
    # ... sorted by ranking_score descending
]
```

---

## 7. Evaluation Methods

### Offline Metrics (Per Task)

| Task | Metric | Description | Target |
|------|--------|-------------|--------|
| P(click) | **AUC-ROC** | Discriminative power for click prediction | > 0.78 |
| P(click) | **Log Loss** | Calibration quality | < 0.35 |
| E[watch_time] | **RMSE** | Watch time prediction accuracy | Directional |
| E[watch_time] | **R-squared** | Variance explained | > 0.25 |
| P(like) | **AUC-ROC** | Like prediction quality | > 0.75 |
| P(finish) | **AUC-ROC** | Completion prediction quality | > 0.72 |
| P(dislike) | **AUC-ROC** | Dislike prediction quality | > 0.70 |
| Combined | **NDCG@20** | End-to-end ranking quality | Primary offline metric |

### Why AUC and Not Accuracy?

- Labels are heavily imbalanced (CTR ~3-5%, like rate ~1-2%)
- AUC measures ranking quality independent of threshold
- Log loss measures calibration (important for score combination)

### Offline Evaluation Code

```python
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
import numpy as np


def evaluate_ranking_model(model, test_loader, device):
    """Comprehensive offline evaluation of the multi-task ranking model."""
    model.eval()

    all_preds = {task: [] for task in ["click", "watch_time", "like", "finish", "dislike"]}
    all_labels = {task: [] for task in all_preds}
    all_clicked = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            preds = model(features)

            for task in all_preds:
                all_preds[task].append(preds[task].cpu().numpy())
                all_labels[task].append(batch[f"label_{task}"].numpy())
            all_clicked.append(batch["label_click"].numpy().flatten())

    # Concatenate
    for task in all_preds:
        all_preds[task] = np.concatenate(all_preds[task]).flatten()
        all_labels[task] = np.concatenate(all_labels[task]).flatten()
    clicked_mask = np.concatenate(all_clicked).astype(bool)

    metrics = {}

    # Click: evaluate on ALL impressions
    metrics["click_auc"] = roc_auc_score(all_labels["click"], all_preds["click"])
    metrics["click_logloss"] = log_loss(all_labels["click"], all_preds["click"])

    # Post-click tasks: evaluate ONLY on clicked impressions
    if clicked_mask.any():
        for task in ["like", "finish", "dislike"]:
            y_true = all_labels[task][clicked_mask]
            y_pred = all_preds[task][clicked_mask]
            if len(np.unique(y_true)) > 1:
                metrics[f"{task}_auc"] = roc_auc_score(y_true, y_pred)

        wt_true = all_labels["watch_time"][clicked_mask]
        wt_pred = all_preds["watch_time"][clicked_mask]
        metrics["watch_time_rmse"] = np.sqrt(mean_squared_error(wt_true, wt_pred))

    # NDCG@K for combined ranking
    metrics["ndcg@20"] = compute_ndcg(all_preds, all_labels, k=20)

    return metrics


def compute_ndcg(preds, labels, k=20,
                 weights={"click": 0.1, "watch_time": 1.0, "like": 0.5,
                          "finish": 0.3, "dislike": -2.0}):
    """
    Compute NDCG@K using the combined ranking score.
    Relevance = actual engagement value (watch_time as primary).
    """
    # Combined predicted score
    pred_scores = sum(weights[t] * preds[t] for t in weights)
    # Ground truth relevance (using watch_time as the primary relevance signal)
    relevance = labels["watch_time"]

    # Group by query (user-session), compute NDCG per group, average
    # Simplified: treat entire test set as one ranking
    from sklearn.metrics import ndcg_score
    return ndcg_score([relevance], [pred_scores], k=k)
```

### Online A/B Test Metrics

| Metric | Description | Minimum Detectable Effect |
|--------|-------------|--------------------------|
| **Total watch time per user/day** | Primary engagement metric | +0.5% |
| **Videos watched per session** | Session depth | +1% |
| **Like rate** | Quality signal | +2% |
| **Dislike rate** | Negative signal (should decrease) | -5% |
| **DAU retention (7-day)** | Long-term health | +0.1% |

### Ablation Studies to Run

1. **Single-task vs. multi-task**: Train separate models per task vs. MMoE
2. **Number of experts**: 4 vs. 6 vs. 8
3. **With/without cross features**: Measure lift from interaction features
4. **Score weight sensitivity**: Vary the combination weights and measure NDCG

---

## 8. Interview Talking Points

1. **Why MMoE over shared-bottom?**
   - Shared-bottom forces all tasks through the same representation → negative transfer when tasks conflict
   - MMoE gates allow each task to use different expert combinations
   - Google's paper showed MMoE especially helps when task correlations are low

2. **Selection bias is critical**
   - Post-click labels only exist if the user clicked
   - Naive: treat non-clicked as "0 watch time" → biases model to predict high CTR items
   - Correct: mask post-click losses to only clicked impressions
   - Advanced: inverse propensity scoring, causal modeling

3. **Position bias correction**
   - Items shown at position 1 get more clicks regardless of relevance
   - Solutions: (a) include position as feature during training, remove at inference, (b) inverse propensity weighting by position, (c) shallow tower for position bias (YouTube 2019)
   - YouTube's approach: train a shallow tower for position bias, subtract it at serving time

4. **Score combination weight tuning**
   - Cannot be tuned by offline metrics alone (no single "correct" answer)
   - Requires online A/B testing with different weight configurations
   - Some teams use Bayesian optimization over weight space

5. **Feature freshness matters enormously**
   - Real-time features (last 20 videos watched) change every minute
   - Stale features → stale predictions → poor user experience
   - Feature store with streaming updates is critical infrastructure

6. **Model serving latency budget**
   - ~300 candidates × model inference < 50ms total
   - Solutions: batch inference, GPU serving, model distillation, feature caching
   - YouTube: uses a smaller model for initial scoring, then a larger model for top-50

7. **Calibration vs. discrimination**
   - AUC measures discrimination (ranking quality)
   - Calibration measures whether P(click)=0.1 means 10% of those items get clicked
   - Calibration matters because we combine predictions across tasks with fixed weights
   - If P(click) is poorly calibrated, the weight tuning becomes meaningless
