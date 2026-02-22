# Gradient Boosted Trees (XGBoost / LightGBM)

## 1. What Are Gradient Boosted Trees?

Gradient Boosted Trees (GBT) build an ensemble of **weak decision trees**, where each tree corrects the mistakes of all previous trees. The final prediction is the sum of all trees' outputs.

```
Single decision tree: weak, high bias, fast
  "Is watch_time > 60s?"
     Yes → "Is like_ratio > 0.8?"
              Yes → score = 0.9
              No  → score = 0.6
     No  → score = 0.2

Gradient boosted trees: 500 weak trees, each fixing the last one's errors

  Tree 1:  rough prediction         → residual error = actual - prediction
  Tree 2:  predicts residual error  → smaller residual
  Tree 3:  predicts remaining error → even smaller residual
  ...
  Tree 500: fixes tiny remaining errors

  Final prediction = Σ (learning_rate × tree_i prediction)
```

**Key insight**: Each tree doesn't try to predict the target directly. It predicts the **gradient of the loss** (the direction the prediction needs to move). Hence "gradient" boosting.

---

## 2. How Training Works (Step by Step)

### The Algorithm

```
1. Start with a constant prediction (e.g., mean of all labels)
   F₀(x) = mean(y) = 0.3

2. For each round t = 1, 2, ..., T:
   a. Compute gradients: gᵢ = ∂L/∂F(xᵢ)  for each training example
      "How wrong is our current prediction, and in which direction?"

   b. Fit a new decision tree to the gradients
      "Build a tree that predicts which direction to adjust"

   c. Update the model:
      F_t(x) = F_{t-1}(x) + η × tree_t(x)
      η = learning rate (shrinkage), typically 0.01-0.3

3. Final model: F(x) = F₀ + η × tree₁(x) + η × tree₂(x) + ... + η × tree_T(x)
```

### Concrete Example

```
Task: Predict P(click) for (user, video) pairs

Training data:
  Example 1: features=[gaming, mobile, 8pm, short_video], label=1 (clicked)
  Example 2: features=[cooking, desktop, 2pm, long_video], label=0 (no click)
  Example 3: features=[gaming, mobile, 9pm, short_video], label=1 (clicked)

Round 0: Initial prediction for all = 0.5

Round 1:
  Gradients: [+0.5, -0.5, +0.5]  (need to go UP for ex1,ex3; DOWN for ex2)
  Tree 1 learns: "If category=gaming AND device=mobile → +0.4, else → -0.3"
  Updated predictions: [0.5+0.04, 0.5-0.03, 0.5+0.04] = [0.54, 0.47, 0.54]
  (with learning_rate=0.1)

Round 2:
  Gradients: [+0.46, -0.47, +0.46]  (still need adjustment)
  Tree 2 learns: "If hour > 7pm AND duration < 5min → +0.3, else → -0.2"
  Updated predictions: [0.54+0.03, 0.47-0.02, 0.54+0.03] = [0.57, 0.45, 0.57]

... repeat 500 times ...

Final predictions: [0.89, 0.12, 0.91]  (close to true labels [1, 0, 1])
```

---

## 3. XGBoost vs. LightGBM vs. CatBoost

Three major implementations. All are gradient boosted trees; they differ in how they build each tree.

### XGBoost (eXtreme Gradient Boosting)

```
How it builds each tree:
  - Level-wise growth: grows the tree level by level (all nodes at same depth)
  - Exact greedy split finding: tries all possible split points
  - Regularization: L1 and L2 penalties on leaf weights

  Level-wise growth:
    Level 0:    [root]
    Level 1:    [left]  [right]          ← grow ALL nodes at this level
    Level 2:    [LL] [LR] [RL] [RR]     ← then ALL nodes at this level
```

### LightGBM (Light Gradient Boosting Machine)

```
Key differences from XGBoost:
  - Leaf-wise growth: grows the leaf with the highest loss reduction first
  - Histogram-based split finding: bin continuous features into 256 buckets
  - GOSS (Gradient-based One-Side Sampling): keep all large-gradient examples,
    subsample small-gradient ones → faster training
  - EFB (Exclusive Feature Bundling): bundle mutually exclusive sparse features

  Leaf-wise growth:
    Step 0:    [root]
    Step 1:    [left]  [right]           ← split the node with highest gain
    Step 2:    [left]  [RL] [RR]         ← only split the BEST leaf next
    Step 3:    [LL] [LR] [RL] [RR]      ← keep splitting best leaf

  Leaf-wise is faster and often more accurate, but risks overfitting
  on small datasets (deep trees on one side).
```

### CatBoost (Categorical Boosting)

```
Key differences:
  - Native categorical feature handling (no manual one-hot encoding)
  - Ordered boosting: uses a permutation trick to prevent target leakage
  - Symmetric trees: all leaves at the same depth use the same split features
  - Generally best out-of-the-box without tuning
```

### Comparison

| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| Tree growth | Level-wise | **Leaf-wise** | Symmetric |
| Speed (large data) | Medium | **Fastest** | Slowest |
| Accuracy (default) | Good | Good | **Best** |
| Categorical features | Manual encoding | Basic support | **Native** |
| GPU support | Yes | Yes | Yes |
| Memory usage | Medium | **Lowest** | Highest |
| Overfitting risk | Low | Medium (leaf-wise) | **Lowest** |
| **Best for** | Balanced default | **Large datasets, speed** | Lots of categorical features |

**In recommendations**: LightGBM is most common due to speed on large datasets. CatBoost is popular when features are heavily categorical (user_country, video_category, device_type).

---

## 4. Code: Training with XGBoost and LightGBM

### XGBoost

```python
import xgboost as xgb
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

# ──────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────

# Features: (N, num_features) — mix of numeric and encoded categorical
# Label: binary (clicked or not)
X_train = np.load("train_features.npy")    # (1_000_000, 50)
y_train = np.load("train_labels.npy")      # (1_000_000,) — 0 or 1
X_val = np.load("val_features.npy")        # (200_000, 50)
y_val = np.load("val_labels.npy")          # (200_000,)

# XGBoost's optimized data structure
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

params = {
    "objective":        "binary:logistic",  # Binary classification → outputs probability
    "eval_metric":      "auc",              # Optimize for AUC
    "max_depth":        6,                  # Max tree depth (prevents overfitting)
    "learning_rate":    0.1,                # Shrinkage per tree
    "subsample":        0.8,                # Use 80% of data per tree (row sampling)
    "colsample_bytree": 0.8,               # Use 80% of features per tree
    "min_child_weight": 10,                 # Min samples in a leaf (regularization)
    "reg_alpha":        0.1,                # L1 regularization on leaf weights
    "reg_lambda":       1.0,                # L2 regularization on leaf weights
    "tree_method":      "hist",             # Histogram-based (fast, like LightGBM)
    "device":           "cuda",             # GPU training (if available)
    "seed":             42,
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,                    # Max number of trees
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=20,               # Stop if val AUC doesn't improve for 20 rounds
    verbose_eval=50,                        # Print every 50 rounds
)

# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

y_pred = model.predict(dval)  # Probability scores (0 to 1)
print(f"Val AUC: {roc_auc_score(y_val, y_pred):.4f}")
print(f"Val LogLoss: {log_loss(y_val, y_pred):.4f}")

# Feature importance
importance = model.get_score(importance_type="gain")
top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 features by gain:")
for feat, gain in top_features:
    print(f"  {feat}: {gain:.1f}")

# Save model
model.save_model("click_predictor.xgb")
```

### LightGBM

```python
import lightgbm as lgb

# ──────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────

# LightGBM can handle categorical features natively
# Specify which columns are categorical
categorical_features = ["category", "device_type", "country", "age_bucket"]

train_data = lgb.Dataset(
    X_train, label=y_train,
    categorical_feature=categorical_features,
    free_raw_data=False,
)
val_data = lgb.Dataset(
    X_val, label=y_val,
    categorical_feature=categorical_features,
    reference=train_data,
    free_raw_data=False,
)

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

params = {
    "objective":         "binary",           # Binary classification
    "metric":            "auc",              # Evaluation metric
    "num_leaves":        63,                 # Max leaves per tree (2^6 - 1)
    "learning_rate":     0.05,               # Lower rate + more trees = better generalization
    "feature_fraction":  0.8,                # Column sampling
    "bagging_fraction":  0.8,                # Row sampling
    "bagging_freq":      5,                  # Subsample every 5 iterations
    "min_data_in_leaf":  20,                 # Min samples in a leaf
    "lambda_l1":         0.1,                # L1 regularization
    "lambda_l2":         1.0,                # L2 regularization
    "verbosity":         -1,
    "seed":              42,
    "device":            "gpu",              # GPU training (if available)
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    valid_names=["train", "val"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=30),
        lgb.log_evaluation(period=100),
    ],
)

# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

y_pred = model.predict(X_val)
print(f"Val AUC: {roc_auc_score(y_val, y_pred):.4f}")

# Feature importance
importance = model.feature_importance(importance_type="gain")
feature_names = model.feature_name()
top_idx = np.argsort(importance)[::-1][:10]
print("\nTop 10 features by gain:")
for idx in top_idx:
    print(f"  {feature_names[idx]}: {importance[idx]:.1f}")

# Save model
model.save_model("click_predictor.lgb")
```

---

## 5. Code: Inference (Serving)

```python
# ──────────────────────────────────────────────
# Loading and serving (same for both XGBoost and LightGBM)
# ──────────────────────────────────────────────

# XGBoost inference
import xgboost as xgb

model = xgb.Booster()
model.load_model("click_predictor.xgb")

def predict_xgb(features):
    """
    Score candidates.
    features: np.ndarray of shape (N_candidates, num_features)
    returns: np.ndarray of shape (N_candidates,) — P(click) for each candidate
    """
    dmatrix = xgb.DMatrix(features)
    return model.predict(dmatrix)


# LightGBM inference
import lightgbm as lgb

model = lgb.Booster(model_file="click_predictor.lgb")

def predict_lgb(features):
    """Score candidates."""
    return model.predict(features)


# ──────────────────────────────────────────────
# Serving characteristics
# ──────────────────────────────────────────────

"""
Latency for 300 candidates:
  XGBoost (500 trees, depth 6):     ~0.5ms on CPU
  LightGBM (1000 trees, 63 leaves): ~0.3ms on CPU

NO GPU NEEDED for inference — trees are CPU-native and extremely fast.
This is a major advantage over neural networks.

Memory:
  XGBoost model (500 trees):  ~5-50 MB
  LightGBM model (1000 trees): ~5-50 MB
  (Compare to neural network: 50-500 MB, requires GPU)
"""
```

---

## 6. Where GBTs Excel (and Where They Don't)

### Where GBTs Are the Best Choice

```
1. TABULAR DATA (structured features)
   - User demographics, item statistics, engagement metrics
   - GBTs are STILL the best model family for tabular data in 2024
   - They outperform neural networks on tabular tasks in most benchmarks

2. RANKING BASELINES
   - Before building a complex MMoE neural network, start with LightGBM
   - Often 80-90% of the neural network's quality with 10% of the effort
   - Many production systems still use GBTs for ranking

3. CTR PREDICTION (CLICK-THROUGH RATE)
   - Ad click prediction (Google, Facebook early systems)
   - Product recommendation ranking (e-commerce)
   - Search result ranking

4. FEATURE-RICH, LOW-EMBEDDING SCENARIOS
   - When features are engineered numeric/categorical, not embeddings
   - When you don't have enough data for embedding-heavy neural models

5. RAPID ITERATION
   - Training: minutes to hours (vs. hours to days for neural nets)
   - No GPU needed
   - Easy to debug (feature importance is built in)
```

### Where Neural Networks Beat GBTs

```
1. SEQUENCE DATA
   - User watch history as a sequence → needs RNN/Transformer
   - GBTs can only use pre-aggregated stats (avg, count) of sequences

2. EMBEDDING-HEAVY TASKS
   - Two-Tower retrieval (learned embeddings in shared space)
   - GBTs can't learn embeddings end-to-end

3. MULTI-TASK LEARNING
   - GBTs are single-task — you'd train 5 separate models
   - Neural MMoE trains one model for 5 tasks (shared knowledge)

4. VERY LARGE DATASETS (10B+ examples)
   - Neural networks scale better with data due to gradient-based
     mini-batch training
   - GBTs need to touch all data for each tree split

5. RAW INPUT (images, text)
   - GBTs need pre-extracted features
   - Neural networks can learn from raw inputs
```

### The Typical Production Pattern

```
Phase 1 (MVP): LightGBM ranking model
  - Fast to build, easy to iterate
  - Good baseline to measure improvements against

Phase 2 (Scale): Neural network (DNN or MMoE) ranking model
  - Once you have enough data and engineering resources
  - 5-15% improvement over GBT baseline in engagement metrics

Phase 3 (Mature): GBT as a FEATURE for the neural network
  - Train a GBT, extract its leaf indices
  - Use leaf indices as categorical features in the neural network
  - Facebook's 2014 paper: "Practical Lessons from Predicting Clicks"
  - The GBT captures non-linear feature interactions that help the NN
```

---

## 7. GBTs as Features (Facebook's Approach)

Facebook's influential 2014 paper showed that GBT + neural network is better than either alone:

```python
# Step 1: Train a GBT model
gbm = lgb.train(params, train_data, num_boost_round=500)

# Step 2: Extract leaf indices (which leaf each example lands in, per tree)
leaf_indices = gbm.predict(X_train, pred_leaf=True)
# Shape: (N_train, 500) — for each example, which leaf in each of the 500 trees

# leaf_indices[0] = [12, 5, 31, 7, ...]
# Meaning: example 0 landed in leaf 12 of tree 0, leaf 5 of tree 1, etc.

# Step 3: Use leaf indices as categorical features in a neural network
# Each leaf index becomes an embedding lookup
# The neural network learns from both raw features AND GBT-discovered patterns

class GBTEnhancedNN(nn.Module):
    def __init__(self, raw_feat_dim, num_trees, num_leaves_per_tree, leaf_emb_dim=8):
        super().__init__()
        # One embedding table per tree
        self.leaf_embeddings = nn.ModuleList([
            nn.Embedding(num_leaves_per_tree, leaf_emb_dim)
            for _ in range(num_trees)
        ])
        combined_dim = raw_feat_dim + num_trees * leaf_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, raw_features, leaf_indices):
        leaf_embs = [
            self.leaf_embeddings[t](leaf_indices[:, t])
            for t in range(len(self.leaf_embeddings))
        ]
        leaf_concat = torch.cat(leaf_embs, dim=1)
        combined = torch.cat([raw_features, leaf_concat], dim=1)
        return torch.sigmoid(self.mlp(combined))
```

---

## 8. Key Hyperparameters Explained

| Parameter | XGBoost Name | LightGBM Name | Effect | Typical Range |
|-----------|-------------|---------------|--------|---------------|
| **Number of trees** | `num_boost_round` | `num_boost_round` | More trees → lower bias, risk overfit | 100-2000 (use early stopping) |
| **Learning rate** | `learning_rate` | `learning_rate` | Lower → needs more trees, better generalization | 0.01-0.3 |
| **Tree depth / leaves** | `max_depth` | `num_leaves` | Deeper → more complex patterns, risk overfit | depth: 4-8, leaves: 31-127 |
| **Row sampling** | `subsample` | `bagging_fraction` | Use subset of data per tree (regularization) | 0.7-0.9 |
| **Column sampling** | `colsample_bytree` | `feature_fraction` | Use subset of features per tree | 0.6-0.9 |
| **Min samples per leaf** | `min_child_weight` | `min_data_in_leaf` | Minimum samples to create a new leaf | 5-50 |
| **L1 regularization** | `reg_alpha` | `lambda_l1` | Sparsity on leaf weights | 0-1.0 |
| **L2 regularization** | `reg_lambda` | `lambda_l2` | Smoothness on leaf weights | 0-10.0 |

### Quick Tuning Strategy

```
1. Fix learning_rate = 0.1, num_boost_round = 1000, early_stopping_rounds = 30
2. Tune tree structure: max_depth (or num_leaves) and min_child_weight
3. Tune sampling: subsample and colsample_bytree
4. Tune regularization: reg_alpha and reg_lambda
5. Lower learning_rate to 0.01-0.05, increase num_boost_round proportionally
6. Use Optuna or Bayesian optimization for automated tuning
```

---

## 9. GBT vs. Neural Network: Head-to-Head

| Dimension | GBT (LightGBM) | Neural Network (MMoE) |
|-----------|----------------|----------------------|
| **Tabular data quality** | Best | Good |
| **Sequence data** | Cannot (need pre-aggregation) | Native (RNN/Transformer) |
| **Embedding learning** | Cannot | Native |
| **Multi-task** | Separate models per task | One model, shared knowledge |
| **Training speed** | Minutes (CPU) | Hours (GPU) |
| **Inference speed** | ~0.3ms CPU | ~5ms GPU |
| **Inference hardware** | CPU only | GPU recommended |
| **Model size** | 5-50 MB | 50-500 MB |
| **Data efficiency** | Better with small data (< 1M) | Better with large data (> 10M) |
| **Feature engineering** | Critical (model depends on it) | Less critical (can learn representations) |
| **Interpretability** | Good (feature importance, SHAP) | Poor (black box) |
| **Infrastructure** | Simple (pip install, CPU) | Complex (GPU, model servers) |
| **Handling missing values** | Native (XGBoost/LightGBM) | Need imputation |

### Decision Framework

```
Use GBT when:
  ✓ Tabular features (structured data)
  ✓ Small-medium data (< 10M examples)
  ✓ Single-task prediction
  ✓ Need fast iteration and debugging
  ✓ No GPU infrastructure
  ✓ Building v1 / baseline

Use Neural Network (MMoE) when:
  ✓ Sequence features (watch history)
  ✓ Large data (> 10M examples)
  ✓ Multiple correlated tasks
  ✓ Need to learn embeddings
  ✓ GPU infrastructure available
  ✓ Building v2+ after GBT baseline
```

---

## 10. Applied Scenarios in Recommendations

### Scenario 1: Ranking Baseline

```python
# The first ranking model at a startup is almost always GBT

features = [
    # User features (pre-aggregated)
    "user_avg_watch_time_7d",       # numeric
    "user_num_videos_7d",           # numeric
    "user_top_category",            # categorical
    "user_device",                  # categorical

    # Item features
    "video_views_log",              # numeric
    "video_completion_rate",        # numeric
    "video_like_ratio",             # numeric
    "video_category",               # categorical
    "video_duration_sec",           # numeric
    "video_age_days",               # numeric

    # User-item interaction features (MOST IMPORTANT)
    "user_watched_this_creator",    # binary
    "user_category_watch_fraction", # numeric: what % of user's watches are this category
    "two_tower_similarity_score",   # numeric: dot(user_emb, item_emb) from retrieval model
]

# Train LightGBM to predict P(click)
# → This alone is a surprisingly strong ranking model
```

### Scenario 2: Feature Interaction Discovery

```python
# GBTs automatically discover feature interactions

# Example tree split:
#   IF video_category == "gaming"
#     AND user_device == "mobile"
#     AND hour_of_day > 18
#     THEN P(click) = 0.82
#
# The model discovered: "mobile users click gaming videos in the evening"
# You didn't need to manually create this cross-feature.

# Neural networks CAN learn this too, but GBTs find it more reliably
# with less data.
```

### Scenario 3: Ad Click Prediction

```
Google's early ad click prediction system was GBTs.
Facebook's ad system combined GBTs with logistic regression.

Ad ranking features:
  - Ad creative features (text length, image type)
  - User features (demographics, browsing history)
  - Context features (page type, ad position)
  - Advertiser features (bid, budget, historical CTR)

GBTs handle the mix of numeric and categorical features naturally.
The built-in feature importance helps explain to advertisers WHY
their ad ranks where it does.
```

---

## 11. Interview Talking Points

### "Why would you use GBT instead of a neural network for ranking?"

> GBTs are still the best model family for tabular/structured features. They train in minutes on CPU, require no GPU infrastructure, handle missing values natively, and provide built-in feature importance for debugging. For a ranking baseline or when data is limited (< 10M examples), GBT often matches or beats a neural network with 10% of the engineering effort.

### "When do you move from GBT to neural networks?"

> When you need sequence modeling (user watch history), embedding learning (Two-Tower retrieval), or multi-task learning (predicting click + watch_time + like simultaneously). The inflection point is typically when you have > 10M training examples, GPU infrastructure, and need the shared representation learning that multi-task neural networks provide.

### "Explain the difference between XGBoost and LightGBM."

> Both are gradient boosted trees. XGBoost grows trees level-by-level (balanced trees), while LightGBM grows leaf-by-leaf (picks the leaf with highest gain next). LightGBM also uses histogram binning for faster split finding. In practice, LightGBM is 2-10x faster with similar or better accuracy, especially on large datasets. XGBoost is more mature and has broader platform support.

### "How does a GBT handle categorical features?"

> XGBoost requires manual encoding (one-hot or label encoding). LightGBM has basic native support (finds optimal splits on categorical values). CatBoost has the best native categorical handling using ordered target statistics. In practice, for recommendation features with high cardinality (user_id, video_id), you wouldn't use raw IDs — you'd use pre-computed embeddings or aggregated statistics as numeric features instead.

### "What's Facebook's GBT + NN approach?"

> Train a GBT first, extract the leaf indices (which leaf each example lands in per tree), and use those as categorical features in a neural network via embedding lookups. The GBT captures non-linear feature interactions (like "gaming + mobile + evening = high CTR") that the neural network can then leverage alongside its own learned representations. This was published in Facebook's 2014 "Practical Lessons" paper and is still used in some ad systems.
