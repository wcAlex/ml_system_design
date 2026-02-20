# Mixture of Experts (MoE / MMoE)

## 1. What Is Mixture of Experts?

Mixture of Experts is a model architecture where multiple "expert" sub-networks each learn different patterns, and a **gating network** learns which experts to use for each input.

```
Standard single model:
  input → [One Big Network] → output
  Every input uses the SAME weights.

Mixture of Experts:
  input → [Expert 1] ─┐
  input → [Expert 2] ──┼── Gate(input) selects/blends experts ──→ output
  input → [Expert 3] ──┤
  input → [Expert 4] ─┘

  Different inputs activate different expert combinations.
  Expert 1 might specialize in "short gaming videos"
  Expert 3 might specialize in "long educational content"
```

**Key insight**: Instead of forcing one network to handle everything, let specialized sub-networks handle different patterns. The gate learns to route each input to the right specialists.

---

## 2. Why MoE Exists: The Multi-Task Problem

### The Problem: Task Interference

When you train one model to predict multiple things (click, watch time, like), the shared layers are pulled in conflicting directions:

```
Shared-bottom multi-task model (NAIVE):

  features → [Shared MLP] → head_1 → P(click)
                           → head_2 → E[watch_time]

Problem: The shared layers must serve BOTH tasks.

  Click prediction wants features about: thumbnail appeal, title curiosity
  Watch time wants features about:       content quality, topic match

  A clickbait video:
    - Has high thumbnail appeal → click task wants shared layers to capture this
    - Has low content quality   → watch time task wants shared layers to ignore this

  The shared layers are pulled in opposite directions → BOTH tasks get worse.
```

### The Solution: MMoE (Multi-gate Mixture of Experts)

Give each task its own gate to select which expert outputs to use:

```
MMoE architecture:

  features
     │
  ┌──┼──┬──┬──┐
  │  │  │  │  │
  E₁ E₂ E₃ E₄    ← Expert networks (each is a small MLP)
  │  │  │  │
  └──┼──┴──┘
     │
  ┌──┴──┐
  │     │
  G₁   G₂          ← Per-task gates (learned softmax over expert outputs)
  │     │
  T₁   T₂          ← Task-specific towers
  │     │
P(click)  E[watch_time]

Gate 1 (click): "Use 60% E₁ + 30% E₃ + 10% E₂"
Gate 2 (watch): "Use 50% E₂ + 40% E₄ + 10% E₁"

Each task gets a DIFFERENT blend of expert knowledge.
No more conflict.
```

---

## 3. Architecture Variants

### Variant 1: Standard MoE (Single Task)

All experts feed into one gate for one task. Used to increase model capacity without proportionally increasing computation (sparse MoE).

```
  features → Expert 1 ─┐
  features → Expert 2 ──┼── Gate → weighted sum → output
  features → Expert 3 ─┘

Gate output: softmax([w₁, w₂, w₃]) → [0.6, 0.3, 0.1]
Result: 0.6 × E₁_output + 0.3 × E₂_output + 0.1 × E₃_output
```

### Variant 2: MMoE (Multi-gate, Multi-Task) — Most Common in Recommendations

Each task has its own gate. Experts are shared but gated differently per task.

Published by Google in 2018. Used at YouTube, Google Ads, and across the industry.

### Variant 3: PLE (Progressive Layered Extraction)

Extension of MMoE from Tencent (2020). Adds **task-specific experts** in addition to shared experts:

```
PLE:
  Shared experts:     E_s1, E_s2, E_s3     (used by all tasks)
  Task 1 experts:     E_1a, E_1b            (used only by task 1)
  Task 2 experts:     E_2a, E_2b            (used only by task 2)

  Gate 1 selects from: E_s1, E_s2, E_s3, E_1a, E_1b
  Gate 2 selects from: E_s1, E_s2, E_s3, E_2a, E_2b

  Stacked across multiple extraction layers.
```

**PLE > MMoE** when tasks are very different (e.g., click vs. comment). Adds more parameters but reduces interference further.

### Variant 4: Sparse MoE (Used in LLMs)

Only activate a **subset** of experts per input (top-k gating). Used in large language models (Switch Transformer, Mixtral) to scale model size without scaling compute.

```
Sparse MoE (top-2 gating):
  Gate output: [0.05, 0.60, 0.02, 0.33]
  Top-2: Expert 2 (0.60) and Expert 4 (0.33)

  Only run Expert 2 and Expert 4 — skip Expert 1 and Expert 3.
  → 2/4 = 50% compute savings with most of the capacity.
```

### Comparison

| Variant | Tasks | Expert Sharing | Compute | Used At |
|---------|-------|---------------|---------|---------|
| Standard MoE | 1 | All shared | Full (dense) or partial (sparse) | LLMs (Mixtral) |
| **MMoE** | Multiple | All shared, per-task gates | Full | **YouTube, Google Ads** |
| PLE | Multiple | Shared + task-specific | Full | Tencent, TikTok |
| Sparse MoE | 1 or multiple | Top-k routing | Reduced | GPT-4, Switch Transformer |

---

## 4. Code: MMoE from Scratch (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """
    A single expert network.

    Each expert is a small MLP that learns to specialize
    in a different aspect of the input.
    """

    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Gate(nn.Module):
    """
    Gate network for one task.

    Takes the raw input features and outputs a softmax distribution
    over experts, indicating how much each expert should contribute
    to this task.
    """

    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.gate(x)  # (B, num_experts)


class TaskTower(nn.Module):
    """
    Task-specific tower that produces the final prediction.

    Takes the gated expert output and produces a scalar prediction
    (e.g., P(click) or E[watch_time]).
    """

    def __init__(self, input_dim, hidden_dim=128, output_dim=1):
        super().__init__()
        self.tower = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.tower(x)


class MMoE(nn.Module):
    """
    Multi-gate Mixture of Experts (Google, 2018).

    Architecture:
        Input → N experts (shared)
              → K gates (one per task, learned routing)
              → K towers (one per task, final prediction)

    This is ONE model — all components are trained jointly end-to-end.
    At serving time, it runs as a single forward pass producing all task predictions.

    Args:
        input_dim:    Dimension of input features
        num_experts:  Number of expert networks
        expert_dim:   Output dimension of each expert
        task_configs: List of dicts with task configuration
                      e.g., [{"name": "click", "type": "binary"},
                             {"name": "watch_time", "type": "regression"}]
    """

    def __init__(self, input_dim, num_experts=6, expert_dim=128, task_configs=None):
        super().__init__()

        if task_configs is None:
            task_configs = [
                {"name": "click", "type": "binary"},
                {"name": "watch_time", "type": "regression"},
                {"name": "like", "type": "binary"},
                {"name": "finish", "type": "binary"},
                {"name": "dislike", "type": "binary"},
            ]

        self.num_experts = num_experts
        self.task_configs = task_configs
        self.num_tasks = len(task_configs)

        # Shared experts
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim=256, output_dim=expert_dim)
            for _ in range(num_experts)
        ])

        # Per-task gates
        self.gates = nn.ModuleList([
            Gate(input_dim, num_experts)
            for _ in range(self.num_tasks)
        ])

        # Per-task towers
        self.towers = nn.ModuleList([
            TaskTower(expert_dim, hidden_dim=128, output_dim=1)
            for _ in range(self.num_tasks)
        ])

    def forward(self, features):
        """
        Single forward pass produces ALL task predictions.

        Args:
            features: (B, input_dim) concatenated user + item + context features

        Returns:
            dict of {task_name: (B, 1) predictions}
        """
        # Step 1: Run ALL experts on the input
        expert_outputs = [expert(features) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (B, num_experts, expert_dim)

        # Step 2: For each task, compute gated expert combination
        predictions = {}

        for task_idx, task_config in enumerate(self.task_configs):
            # Gate: (B, num_experts) — how much weight for each expert
            gate_weights = self.gates[task_idx](features)  # (B, num_experts)

            # Weighted sum of expert outputs
            # gate_weights: (B, num_experts, 1) × expert_outputs: (B, num_experts, expert_dim)
            gated_output = torch.sum(
                gate_weights.unsqueeze(2) * expert_outputs,
                dim=1
            )  # (B, expert_dim)

            # Task tower: (B, expert_dim) → (B, 1)
            raw_pred = self.towers[task_idx](gated_output)

            # Apply appropriate activation
            if task_config["type"] == "binary":
                pred = torch.sigmoid(raw_pred)
            else:
                pred = F.relu(raw_pred)  # Non-negative for regression

            predictions[task_config["name"]] = pred

        return predictions

    def get_gate_weights(self, features):
        """
        Inspect which experts each task is using (for debugging/interpretability).
        """
        with torch.no_grad():
            gate_weights = {}
            for task_idx, task_config in enumerate(self.task_configs):
                weights = self.gates[task_idx](features)  # (B, num_experts)
                gate_weights[task_config["name"]] = weights.mean(dim=0).cpu().numpy()
            return gate_weights
```

---

## 5. Code: Training Loop

```python
class MultiTaskLoss(nn.Module):
    """
    Combined loss for all tasks.

    Key detail: Post-click signals (like, finish, dislike) can only be observed
    for videos that were actually clicked. We mask the loss for unclicked items
    to avoid training on missing labels.
    """

    def __init__(self, task_configs):
        super().__init__()
        self.task_configs = task_configs
        self.bce = nn.BCELoss(reduction="none")
        self.huber = nn.HuberLoss(reduction="none", delta=10.0)

    def forward(self, predictions, labels, clicked_mask=None):
        """
        Args:
            predictions: dict of {task_name: (B, 1)}
            labels:      dict of {task_name: (B, 1)}
            clicked_mask: (B, 1) — 1 if user clicked, 0 otherwise
                          Used to mask post-click losses
        """
        total_loss = 0
        task_losses = {}

        for task_config in self.task_configs:
            name = task_config["name"]
            pred = predictions[name]
            label = labels[name]

            if task_config["type"] == "binary":
                loss = self.bce(pred, label)
            else:
                loss = self.huber(pred, label)

            # Post-click signals: only compute loss where the item was clicked
            if name in ("like", "finish", "dislike") and clicked_mask is not None:
                loss = loss * clicked_mask

            loss = loss.mean()
            task_losses[name] = loss
            total_loss += loss

        return total_loss, task_losses


def train_mmoe(model, train_loader, val_loader, device, epochs=10, lr=1e-3):
    """Full training loop for MMoE ranking model."""

    task_configs = model.task_configs
    criterion = MultiTaskLoss(task_configs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.to(device)

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        epoch_losses = {tc["name"]: 0.0 for tc in task_configs}
        num_batches = 0

        for batch in train_loader:
            features = batch["features"].to(device)

            labels = {
                tc["name"]: batch[tc["name"]].to(device)
                for tc in task_configs
            }
            clicked_mask = batch.get("clicked_mask")
            if clicked_mask is not None:
                clicked_mask = clicked_mask.to(device)

            predictions = model(features)
            total_loss, task_losses = criterion(predictions, labels, clicked_mask)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for name, loss in task_losses.items():
                epoch_losses[name] += loss.item()
            num_batches += 1

        scheduler.step()

        # ── Validate ──
        model.eval()
        val_preds = {tc["name"]: [] for tc in task_configs}
        val_labels = {tc["name"]: [] for tc in task_configs}

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                predictions = model(features)

                for tc in task_configs:
                    val_preds[tc["name"]].append(predictions[tc["name"]].cpu())
                    val_labels[tc["name"]].append(batch[tc["name"]])

        # Compute per-task AUC
        from sklearn.metrics import roc_auc_score

        auc_str = ""
        for tc in task_configs:
            if tc["type"] == "binary":
                preds = torch.cat(val_preds[tc["name"]]).numpy().flatten()
                labels = torch.cat(val_labels[tc["name"]]).numpy().flatten()
                try:
                    auc = roc_auc_score(labels, preds)
                    auc_str += f" {tc['name']}_AUC={auc:.4f}"
                except ValueError:
                    auc_str += f" {tc['name']}_AUC=N/A"

        loss_str = " ".join(
            f"{name}={epoch_losses[name]/num_batches:.4f}"
            for name in epoch_losses
        )
        print(f"Epoch {epoch+1}/{epochs}: {loss_str} |{auc_str}")

    return model
```

---

## 6. Code: Inference (Serving)

```python
def rank_candidates(model, candidates_features, task_weights, device):
    """
    Score and rank candidate videos for a user.

    This is ONE forward pass through ONE model.
    All 5 task predictions come out simultaneously.

    Args:
        model:               Trained MMoE model
        candidates_features: (N_candidates, input_dim) features for each candidate
        task_weights:        dict of {task_name: weight} for score combination
                             e.g., {"click": 0.1, "watch_time": 1.0, "like": 0.5,
                                    "finish": 0.3, "dislike": -2.0}

    Returns:
        ranked_indices:  Candidate indices sorted by combined score (best first)
        combined_scores: The combined scores
        per_task_preds:  Individual task predictions for debugging
    """
    model.eval()

    with torch.no_grad():
        features = candidates_features.to(device)
        predictions = model(features)   # ONE forward pass → all 5 predictions

    # Combine predictions into a single ranking score
    combined_scores = torch.zeros(len(candidates_features), device=device)

    for task_name, weight in task_weights.items():
        combined_scores += weight * predictions[task_name].squeeze()

    # Sort by combined score (descending)
    ranked_indices = torch.argsort(combined_scores, descending=True)

    return (
        ranked_indices.cpu().numpy(),
        combined_scores.cpu().numpy(),
        {k: v.cpu().numpy() for k, v in predictions.items()},
    )


# ──────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────

"""
# After retrieval gives us 300 candidate videos:
candidate_features = feature_store.get_features(user_id, candidate_video_ids)
# candidate_features shape: (300, input_dim)

# Score all 300 candidates in ONE forward pass (~5ms on GPU)
ranked_idx, scores, per_task = rank_candidates(
    model=mmoe_model,
    candidates_features=candidate_features,
    task_weights={
        "click":      0.1,
        "watch_time": 1.0,
        "like":       0.5,
        "finish":     0.3,
        "dislike":   -2.0,
    },
    device="cuda",
)

# Top-5 candidates after ranking:
for i in range(5):
    idx = ranked_idx[i]
    print(f"#{i+1}: video={candidate_video_ids[idx]}")
    print(f"      score={scores[idx]:.3f}")
    print(f"      P(click)={per_task['click'][idx][0]:.3f}")
    print(f"      E[watch_time]={per_task['watch_time'][idx][0]:.1f}s")
    print(f"      P(like)={per_task['like'][idx][0]:.3f}")
"""
```

---

## 7. Is It One Model? How Is It Hosted?

**Yes, MMoE is one model.** This is a common point of confusion.

```
Conceptually:  "Mixture of experts" sounds like multiple separate models
Reality:       It's ONE nn.Module with ONE set of parameters, ONE forward pass

Training:   ONE training loop, ONE optimizer, ONE backward pass
Saving:     ONE model checkpoint: torch.save(model.state_dict(), "mmoe.pt")
Serving:    ONE model loaded into memory, ONE forward pass per request
Deployment: ONE Docker container, ONE model server endpoint

There is no separate "expert 1 service" and "expert 2 service."
Everything runs together in a single forward pass.
```

### Serving Architecture

```
Production deployment:

  Model server (e.g., TorchServe, Triton Inference Server):
  ┌─────────────────────────────────────────────┐
  │  GPU Memory:                                 │
  │    mmoe_model (ONE model, ~50-200MB)         │
  │                                              │
  │  Input:  (300, input_dim) candidate features │
  │  Forward: ONE pass through all experts,      │
  │           gates, and towers simultaneously   │
  │  Output: {click: (300,1), watch_time: (300,1),│
  │           like: (300,1), finish: (300,1),     │
  │           dislike: (300,1)}                   │
  │                                              │
  │  Latency: ~5-10ms for 300 candidates on GPU  │
  └─────────────────────────────────────────────┘

The orchestrator sends 300 candidates → gets back 5 predictions each.
Then applies the weight combination OUTSIDE the model (so weights can be
changed without redeploying the model).
```

### Why Not Separate Models?

You might ask: "Why not train 5 separate models, one per task?" Three reasons:

```
1. SHARED KNOWLEDGE: Expert networks learn shared representations that
   help ALL tasks. "This video has high production quality" helps predict
   both watch_time and like. Separate models would each learn this redundantly.

2. EFFICIENCY: ONE forward pass for 5 predictions is much cheaper than
   5 separate forward passes. 5ms vs. 25ms.

3. JOINT OPTIMIZATION: Tasks inform each other during training.
   Click and watch_time predictions are correlated — training them
   together gives better gradients than training in isolation.
```

---

## 8. How Experts Specialize (Intuition)

After training, experts naturally specialize. You can inspect gate weights to see what each expert captures:

```python
# Inspect what experts each task uses
gate_weights = model.get_gate_weights(sample_features)

# Hypothetical output:
# click task:      [0.35, 0.10, 0.30, 0.05, 0.15, 0.05]
# watch_time task: [0.05, 0.40, 0.05, 0.30, 0.10, 0.10]
# like task:       [0.10, 0.30, 0.10, 0.25, 0.15, 0.10]
# dislike task:    [0.05, 0.10, 0.05, 0.10, 0.05, 0.65]

# Interpretation:
# Expert 1: Heavily used by click → likely learns visual/curiosity patterns
# Expert 2: Heavily used by watch_time, like → likely learns content quality
# Expert 4: Used by watch_time, like → likely learns user-content match
# Expert 6: Heavily used by dislike → likely learns negative quality signals
```

**The specialization is emergent** — you don't tell experts what to learn. The gating mechanism and backpropagation naturally push experts toward different specializations because it minimizes the combined loss.

---

## 9. Practical Considerations

### How Many Experts?

```
Rule of thumb: 4-8 experts for recommendation ranking

Too few (2-3): Not enough specialization, behaves like shared-bottom
Too many (16+): Each expert gets less gradient signal, slower training,
                gate has trouble choosing among many options

YouTube/Google typically use 6-8 experts.
```

### Expert Size vs. Number

```
Budget: ~50M total parameters for experts

Option A: 4 large experts (12.5M each)
  Each expert is powerful but less specialized

Option B: 8 medium experts (6.25M each)
  More specialization, each expert is smaller

Option C: 16 small experts (3.1M each)
  Highly specialized but each expert has limited capacity

Best practice: 6-8 medium experts. Each expert is a 2-3 layer MLP.
```

### Load Balancing (Sparse MoE)

In sparse MoE (top-k routing), some experts may get all the traffic while others are idle:

```
Problem: Gate always picks Expert 2 → Expert 2 gets all gradients,
         others stagnate → wastes parameters

Solution: Add a load balancing loss:
  L_balance = α × variance(expert_usage)
  Penalizes uneven expert utilization

This is critical for sparse MoE in LLMs (Switch Transformer, Mixtral).
Less important for dense MMoE in recommendations (all experts are always used).
```

---

## 10. MMoE vs. Shared-Bottom Multi-Head Model

This is a common point of confusion. Both architectures train multiple tasks jointly with shared knowledge. The difference is **how knowledge is shared**.

### What Is a Shared-Bottom Multi-Head Model?

A single shared backbone with multiple task-specific heads (classifiers/regressors). This is the simpler, more common multi-task architecture — used everywhere from multi-label image classification to multi-task NLP.

```
Shared-Bottom Multi-Head:

  features → [Shared MLP layers] → shared_repr (one representation for ALL tasks)
                                       │
                              ┌────────┼────────┐
                              │        │        │
                           [Head 1] [Head 2] [Head 3]
                              │        │        │
                          P(click) E[watch]  P(like)

  Key: ONE shared representation feeds ALL task heads.
  Every task sees the EXACT SAME representation.
```

### Side-by-Side Comparison

```
Shared-Bottom Multi-Head:              MMoE:

  features                              features
     │                                     │
  [Shared MLP]                        [E₁] [E₂] [E₃] [E₄]   ← Multiple experts
     │                                  │    │    │    │
  shared_repr ← ONE repr               └────┼────┴────┘
     │                                      │
  ┌──┼──┐                              ┌───┼───┐
  H₁ H₂ H₃  ← heads                   G₁  G₂  G₃  ← per-task GATES
  │  │  │                               │   │   │
  │  │  │                              T₁  T₂  T₃  ← task towers
  │  │  │                               │   │   │

  All heads see                        Each task gets a
  the SAME repr                        DIFFERENT repr
                                       (custom blend of experts)
```

The fundamental difference in one sentence: **In shared-bottom, all tasks consume the same representation. In MMoE, each task assembles its own representation via gating.**

### Why Does This Matter? A Concrete Example

```
Video: Clickbait cooking video
  - Flashy thumbnail, sensational title
  - Actual content is low quality, users abandon after 10 seconds

Shared-Bottom Multi-Head:
  Shared MLP produces ONE representation for this video.
  This representation must simultaneously help predict:
    - P(click) = HIGH  (flashy thumbnail → users click)
    - E[watch_time] = LOW  (bad content → users leave)

  CONFLICT: The shared layers are pulled to encode "flashy thumbnail = good"
  for click task but "flashy thumbnail ≠ good content" for watch time task.
  Compromise: a muddled representation that's mediocre for both tasks.

  Result:
    P(click) prediction:     0.60 (should be ~0.80, under-predicted)
    E[watch_time] prediction: 45s  (should be ~10s, over-predicted)

MMoE:
  Expert 1 learns: "flashy thumbnail patterns" → high activation
  Expert 2 learns: "content quality signals" → low activation
  Expert 3 learns: "user-content topic match" → medium activation

  Gate for P(click):       [E1=0.7, E2=0.1, E3=0.2]
    → Representation is DOMINATED by Expert 1 (thumbnail patterns)
    → P(click) = 0.82  ✓ accurate

  Gate for E[watch_time]:  [E1=0.1, E2=0.6, E3=0.3]
    → Representation is DOMINATED by Expert 2 (content quality)
    → E[watch_time] = 12s  ✓ accurate

  No conflict — each task gets a custom view of the data.
```

### Code Comparison

```python
# ─── Shared-Bottom Multi-Head ───

class SharedBottomMultiHead(nn.Module):
    """The simpler multi-task architecture. One shared backbone, multiple heads."""

    def __init__(self, input_dim, hidden_dim=256, shared_dim=128, num_tasks=5):
        super().__init__()
        # ONE shared backbone
        self.shared_backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, shared_dim),
            nn.ReLU(),
        )
        # Multiple task heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            for _ in range(num_tasks)
        ])

    def forward(self, features):
        shared_repr = self.shared_backbone(features)  # ONE representation
        return [head(shared_repr) for head in self.heads]  # ALL heads see SAME repr


# ─── MMoE ───

class MMoEModel(nn.Module):
    """Multiple experts + per-task gating. Each task gets a DIFFERENT representation."""

    def __init__(self, input_dim, num_experts=6, expert_dim=128, num_tasks=5):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, expert_dim), nn.ReLU())
            for _ in range(num_experts)
        ])
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, num_experts), nn.Softmax(dim=1))
            for _ in range(num_tasks)
        ])
        self.towers = nn.ModuleList([
            nn.Sequential(nn.Linear(expert_dim, 64), nn.ReLU(), nn.Linear(64, 1))
            for _ in range(num_tasks)
        ])

    def forward(self, features):
        expert_outs = torch.stack([e(features) for e in self.experts], dim=1)  # (B, E, D)
        results = []
        for task_idx in range(len(self.towers)):
            gate_weights = self.gates[task_idx](features)           # (B, E)
            gated = (gate_weights.unsqueeze(2) * expert_outs).sum(1) # (B, D) ← CUSTOM per task
            results.append(self.towers[task_idx](gated))
        return results
```

### When Each Architecture Wins

| Scenario | Shared-Bottom Multi-Head | MMoE |
|----------|------------------------|------|
| Tasks are **highly correlated** (e.g., click & add-to-cart) | Works well — shared repr helps both | Overkill, marginal improvement |
| Tasks **conflict** (e.g., click & watch_time) | Degrades — task interference | **Solves the conflict via gating** |
| Few tasks (2-3) | Simpler, good enough | Small benefit |
| Many tasks (5+) | Interference grows with each task | **Benefit grows with each task** |
| Limited data (< 1M examples) | Better — fewer parameters | May overfit (more parameters) |
| Large data (> 10M examples) | Underfits | **Better utilization of data** |
| Need fast iteration / simplicity | **Simpler to implement and debug** | More moving parts |

### The Progression in Practice

```
Stage 1: Single-task model (predict P(click) only)
  ↓ realize clickbait problem
Stage 2: Shared-bottom multi-head (predict click + watch_time + like)
  ↓ notice task interference (click AUC improves but watch_time gets worse)
Stage 3: MMoE (per-task gating solves the interference)
  ↓ if tasks are VERY different
Stage 4: PLE (add task-specific experts on top of shared experts)
```

Most companies start with shared-bottom multi-head because it's simpler. They move to MMoE when they observe that adding a new task hurts existing tasks — that's the telltale sign of task interference.

### Interview Answer: "What's the difference?"

> Both are multi-task architectures with shared knowledge. Shared-bottom multi-head produces ONE shared representation that all task heads consume — simple, but tasks that need conflicting features interfere with each other. MMoE replaces the single shared backbone with multiple expert networks plus per-task gates, so each task can assemble its OWN custom representation by blending experts differently. The gate is the key difference — it breaks the "one size fits all" bottleneck. In practice, MMoE shows clear gains when tasks conflict (like click vs. watch_time), but shared-bottom is fine when tasks are well-aligned.

---

## 11. MoE in LLMs: A Different Beast

MoE is used in both recommendations (MMoE) and large language models (Mixtral, Switch Transformer, DeepSeek). The core gating idea is the same, but nearly everything else differs.

### 11.1 Why LLMs Use MoE

The problem in LLMs: scaling model size improves quality, but compute and memory scale linearly with parameters.

```
Dense model (standard Transformer):
  Parameters: 70B
  Compute per token: proportional to 70B params
  Memory: must load all 70B params

Sparse MoE model:
  Total parameters: 140B (across all experts)
  Active parameters per token: 20B (only 2 of 8 experts fire)
  Compute per token: proportional to 20B params ← much cheaper
  Memory: must load all 140B params ← still large

Key insight: You get the QUALITY of a 140B model with the COMPUTE of a 20B model.
The catch: you still need the MEMORY for 140B params.
```

### 11.2 Where MoE Lives in a Transformer

In a standard Transformer, each layer has:
1. Self-attention (unchanged in MoE)
2. Feed-Forward Network (FFN) — **this is what gets replaced by MoE**

```
Standard Transformer layer:
  input → [Self-Attention] → [FFN] → output
                               ↑
                          One big FFN
                          (every token uses the same weights)

MoE Transformer layer:
  input → [Self-Attention] → [MoE layer] → output
                               ↑
                          Router picks top-k experts
                          (each token may use DIFFERENT experts)

MoE layer detail:
  ┌─────────────────────────────────────────────┐
  │  Router(token_hidden_state) → [0.1, 0.6, 0.02, 0.28, ...]  │
  │                                                               │
  │  Top-2: Expert 2 (weight=0.6) and Expert 4 (weight=0.28)     │
  │                                                               │
  │  output = 0.6 × Expert_2(token) + 0.28 × Expert_4(token)     │
  │                                                               │
  │  Expert 1, 3, 5, 6, 7, 8 → NOT computed (skipped)            │
  └─────────────────────────────────────────────┘
```

### 11.3 Recommendation MoE vs. LLM MoE: Side-by-Side

```
                     Recommendation (MMoE)          LLM (Sparse MoE)
                     ─────────────────────          ─────────────────
What is routed?      One feature vector per         One TOKEN per routing decision
                     (user, video) pair             (each token in the sequence
                                                     is routed independently)

Dense or sparse?     DENSE: ALL experts run         SPARSE: only top-k (1-2)
                     on every input                 experts run per token

Why MoE?             Multi-task learning            Scale model size without
                     (avoid task interference)      scaling per-token compute

Number of experts    4-8                            8-64 (or more)

Expert size          Small MLP (2-3 layers)         Full FFN layer (~billions of params)

Gate output          Soft weights for ALL experts   Hard top-k selection
                     [0.2, 0.3, 0.1, 0.4]          [0, 0.68, 0, 0.32, 0, 0, 0, 0]

Task heads           Multiple (click, watch, like)  ONE (next token prediction)

Where in model       Replaces the shared backbone   Replaces FFN in each
                     (between input and task heads)  Transformer layer

Training concern     Task interference              Load balancing
                                                    (experts getting equal usage)
```

### 11.4 Key LLM MoE Models

| Model | Year | Experts | Active | Total Params | Active Params | Key Innovation |
|-------|------|---------|--------|-------------|---------------|----------------|
| **Switch Transformer** (Google) | 2021 | 128 | top-1 | 1.6T | 13B | First large-scale sparse MoE; top-1 routing for simplicity |
| **Mixtral 8x7B** (Mistral) | 2023 | 8 | top-2 | 47B | 13B | Open-source; matches Llama 2 70B quality at 13B active compute |
| **DeepSeek-V2** | 2024 | 160 | top-6 | 236B | 21B | Fine-grained experts + shared experts |
| **Mixtral 8x22B** (Mistral) | 2024 | 8 | top-2 | 176B | 44B | Scaled up Mixtral |
| **DBRX** (Databricks) | 2024 | 16 | top-4 | 132B | 36B | Fine-grained experts (smaller, more of them) |

### 11.5 Code: Sparse MoE Layer (LLM-Style)

```python
class SparseMoELayer(nn.Module):
    """
    Sparse Mixture of Experts layer for Transformers.

    Replaces the standard FFN in each Transformer layer.
    Each token is routed to the top-k experts only.
    """

    def __init__(self, hidden_dim, ffn_dim, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Router: projects hidden state to expert scores
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)

        # Expert FFNs (each is a standard Transformer FFN)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_dim),
                nn.GELU(),
                nn.Linear(ffn_dim, hidden_dim),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, hidden_dim) — token hidden states

        Returns:
            output: (batch_size, seq_len, hidden_dim) — MoE output
            router_logits: (batch_size, seq_len, num_experts) — for load balancing loss
        """
        B, S, D = x.shape

        # Step 1: Router computes expert scores for each token
        router_logits = self.router(x)                        # (B, S, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)       # (B, S, num_experts)

        # Step 2: Select top-k experts per token
        top_k_probs, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )  # both (B, S, top_k)

        # Renormalize top-k weights to sum to 1
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Step 3: Compute expert outputs (only for selected experts)
        # Simple implementation: loop over experts, mask tokens
        # (Production uses batched scatter/gather for efficiency)
        output = torch.zeros_like(x)

        for expert_idx in range(self.num_experts):
            # Which tokens selected this expert?
            # mask: (B, S, top_k) → check if any top_k slot picked this expert
            expert_mask = (top_k_indices == expert_idx)  # (B, S, top_k)

            if not expert_mask.any():
                continue  # No tokens routed here → skip entirely

            # Get the weight for this expert (from the matching top_k slot)
            expert_weights = (top_k_probs * expert_mask.float()).sum(dim=-1)  # (B, S)

            # Run the expert on ALL tokens (simpler, slightly wasteful)
            # Production implementations batch only the routed tokens
            expert_out = self.experts[expert_idx](x)  # (B, S, D)

            # Weighted contribution
            output += expert_out * expert_weights.unsqueeze(-1)

        return output, router_logits


class MoETransformerLayer(nn.Module):
    """One Transformer layer with MoE replacing the FFN."""

    def __init__(self, hidden_dim, num_heads, ffn_dim, num_experts=8, top_k=2):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.moe = SparseMoELayer(hidden_dim, ffn_dim, num_experts, top_k)
        self.moe_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, attn_mask=None):
        # Self-attention (unchanged from standard Transformer)
        residual = x
        x = self.attn_norm(x)
        x, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = x + residual

        # MoE FFN (replaces standard FFN)
        residual = x
        x = self.moe_norm(x)
        moe_out, router_logits = self.moe(x)
        x = moe_out + residual

        return x, router_logits
```

### 11.6 The Load Balancing Problem (Critical for LLM MoE)

In recommendation MMoE, all experts run on every input — no balancing needed. In LLM sparse MoE, the router might always pick the same 1-2 experts, leaving others unused:

```
Problem — expert collapse:
  Router always picks Expert 2 and Expert 5
  → Expert 2, 5 get all gradients, become very good
  → Other experts get no gradients, stay random
  → Model effectively uses only 2 of 8 experts
  → Wasted 75% of parameters

Token routing over 1000 tokens:
  Expert 1:  12 tokens    (under-used)
  Expert 2: 380 tokens    ← overloaded
  Expert 3:   5 tokens    (under-used)
  Expert 4:  18 tokens
  Expert 5: 350 tokens    ← overloaded
  Expert 6:   3 tokens    (almost dead)
  Expert 7: 215 tokens
  Expert 8:  17 tokens
```

**Solution**: Add an auxiliary load balancing loss:

```python
def load_balancing_loss(router_logits, top_k_indices, num_experts):
    """
    Encourage equal expert utilization.

    From Switch Transformer (Google, 2021).
    """
    # f_i: fraction of tokens routed to expert i
    # P_i: average router probability for expert i

    router_probs = F.softmax(router_logits, dim=-1)  # (B, S, E)

    # Fraction of tokens assigned to each expert
    one_hot = F.one_hot(top_k_indices, num_experts).float()  # (B, S, top_k, E)
    tokens_per_expert = one_hot.sum(dim=(0, 1, 2))            # (E,)
    f = tokens_per_expert / tokens_per_expert.sum()            # (E,) fractions

    # Average router probability per expert
    P = router_probs.mean(dim=(0, 1))                          # (E,)

    # Loss: dot(f, P) × num_experts
    # Minimized when both f and P are uniform (1/E each)
    loss = num_experts * (f * P).sum()

    return loss

# Total training loss:
# loss = language_model_loss + α × load_balancing_loss
# α is typically 0.01 to 0.1
```

### 11.7 Mixtral 8x7B: Concrete Example

```
Mixtral architecture:
  - 32 Transformer layers
  - 8 experts per MoE layer (each expert = standard 7B FFN)
  - Top-2 routing per token
  - Total parameters: 47B
  - Active parameters per token: ~13B (attention layers + 2 of 8 expert FFNs)

What happens when processing "The cat sat on the":

  Token "The":  Router → Expert 3 (0.55) + Expert 7 (0.45)
  Token "cat":  Router → Expert 1 (0.62) + Expert 5 (0.38)
  Token "sat":  Router → Expert 3 (0.48) + Expert 2 (0.52)
  Token "on":   Router → Expert 7 (0.70) + Expert 4 (0.30)
  Token "the":  Router → Expert 3 (0.51) + Expert 7 (0.49)

  Each token independently picks its own 2 experts.
  Common function words ("the", "on") may converge to similar expert pairs.
  Content words ("cat", "sat") may route to different specialists.
```

### 11.8 Summary: Same Name, Different Usage

| Aspect | Recommendation MMoE | LLM Sparse MoE |
|--------|---------------------|-----------------|
| **Goal** | Multi-task without interference | Scale model size without scaling compute |
| **Routing** | Dense (all experts, soft weights) | Sparse (top-k experts per token) |
| **What's routed** | (user, video) feature vector | Each token independently |
| **Number of tasks** | Multiple (click, watch, like...) | One (next token prediction) |
| **Expert count** | 4-8 | 8-128 |
| **Expert size** | Small MLP (256-d hidden) | Full FFN layer (billions of params) |
| **Key challenge** | Task interference | Load balancing |
| **Where in model** | IS the model backbone | Replaces FFN inside each Transformer layer |
| **Total params** | ~50M | ~50B-1T |

---

## 12. MoE in Other Domains

| Domain | Application | Details |
|--------|------------|---------|
| **LLMs** | Switch Transformer, Mixtral, DeepSeek | Sparse MoE to scale to trillions of parameters |
| **Recommendations** | YouTube, Google Ads ranking | Dense MMoE for multi-task prediction |
| **Computer Vision** | Vision MoE | Expert routing per image patch |
| **Multi-lingual NLP** | Language-specific experts | Gate routes to expert based on detected language |
| **CTR Prediction** | Alibaba, Tencent | PLE (extension of MMoE) |

---

## 13. Interview Talking Points

### "Explain MMoE in one sentence."

> MMoE has multiple shared expert networks and per-task gating networks, so each task can assemble its own custom representation from the experts without interfering with other tasks.

### "Why not just train separate models for each task?"

> Three reasons: (1) Shared experts learn common patterns once instead of redundantly. (2) One forward pass for all predictions is faster than separate passes. (3) Tasks provide complementary gradient signals — click and watch time are correlated, so jointly training improves both.

### "What's the difference between MMoE and PLE?"

> MMoE has only shared experts — every gate can access every expert. PLE adds task-specific experts that only their gate can access. This reduces interference further when tasks are very different (e.g., click vs. comment), but adds more parameters.

### "Is MoE the same thing in recommendations and in LLMs?"

> Same core idea, different usage. In recommendations (MMoE), all experts run on every input (dense routing) for multi-task learning. In LLMs (Switch Transformer, Mixtral), only top-k experts run per token (sparse routing) to scale model size without scaling compute. The gating mechanism is the same; the sparsity is the difference.

### "How do experts specialize?"

> Emergent specialization through backpropagation. You don't manually assign experts to tasks. The gate learns to route inputs to experts that minimize the loss. After training, you can inspect gate weights and typically see that click-heavy tasks favor different experts than watch-time-heavy tasks. The specialization is discovered, not designed.
