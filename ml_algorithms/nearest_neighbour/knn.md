# K-Nearest Neighbors (KNN)

## 1. What Is KNN?

KNN is a **non-parametric, instance-based** algorithm that makes predictions by finding the K closest data points (neighbors) to a query point in the feature space and aggregating their labels.

- **Non-parametric**: No fixed model structure — the "model" *is* the entire training dataset
- **Instance-based (lazy learning)**: No training phase; all computation happens at query time
- **Simple but powerful**: Surprisingly competitive baseline for many tasks

```
Query: "What genre does this user like?"

Feature space (2D example: avg_watch_time, action_movie_ratio):

        action_ratio
        ▲
   1.0  │        ● Action fan
        │      ● Action fan    ★ Query point
   0.7  │        ● Action fan
        │
   0.4  │  ■ Comedy fan
        │    ■ Comedy fan
   0.1  │  ■ Comedy fan
        └──────────────────────────▶ avg_watch_time
             10    30    50    70

K=3: 3 nearest neighbors are all "Action fan" → predict Action fan
```

---

## 2. Algorithm

### Classification (KNN Classifier)

```
INPUT:  Training set D = {(x₁, y₁), ..., (xₙ, yₙ)}, query point q, K
OUTPUT: Predicted class for q

1. Compute distance d(q, xᵢ) for every xᵢ in D
2. Select the K points with the smallest distances
3. Return the majority class among those K neighbors
   (break ties randomly or by distance weight)
```

### Regression (KNN Regressor)

```
Same steps 1-2, then:
3. Return the average (or distance-weighted average) of the K neighbors' values
```

### Distance Metrics

| Metric | Formula | Best For |
|--------|---------|----------|
| **Euclidean** (L2) | `√(Σ(xᵢ - yᵢ)²)` | Continuous features, same scale |
| **Manhattan** (L1) | `Σ|xᵢ - yᵢ|` | High-dimensional, sparse data |
| **Cosine** | `1 - (x·y)/(‖x‖‖y‖)` | Text embeddings, normalized vectors |
| **Minkowski** (Lp) | `(Σ|xᵢ - yᵢ|ᵖ)^(1/p)` | Generalizes L1 (p=1) and L2 (p=2) |

### Choosing K

| K | Behavior | Risk |
|---|----------|------|
| K=1 | Exact nearest neighbor, very sensitive to noise | High variance (overfitting) |
| K=small (3-7) | Good for clean, well-separated data | Some noise sensitivity |
| K=large (20-50) | Smooth decision boundary, robust to noise | High bias (underfitting), slow |
| K=√N | Common rule of thumb starting point | May not be optimal |

**Best practice**: Use cross-validation to select K. Odd K avoids ties in binary classification.

---

## 3. High-Level Code

### From Scratch (NumPy)

```python
import numpy as np
from collections import Counter


class KNNClassifier:
    """K-Nearest Neighbors classifier — exact (brute-force) search."""

    def __init__(self, k: int = 5, metric: str = "euclidean"):
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Store training data. No actual computation happens here."""
        self.X_train = X
        self.y_train = y

    def _compute_distances(self, query: np.ndarray) -> np.ndarray:
        """Compute distance from query to every training point."""
        if self.metric == "euclidean":
            # (N,) array of distances
            return np.sqrt(np.sum((self.X_train - query) ** 2, axis=1))
        elif self.metric == "cosine":
            # Cosine distance = 1 - cosine similarity
            dot = self.X_train @ query
            norms = np.linalg.norm(self.X_train, axis=1) * np.linalg.norm(query)
            return 1 - dot / (norms + 1e-10)
        elif self.metric == "manhattan":
            return np.sum(np.abs(self.X_train - query), axis=1)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def predict_one(self, query: np.ndarray) -> int:
        """Predict class for a single query point."""
        distances = self._compute_distances(query)

        # Find K nearest indices
        k_nearest_idx = np.argpartition(distances, self.k)[:self.k]
        k_nearest_labels = self.y_train[k_nearest_idx]

        # Majority vote
        vote = Counter(k_nearest_labels)
        return vote.most_common(1)[0][0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes for multiple query points."""
        return np.array([self.predict_one(x) for x in X])


class KNNRegressor:
    """K-Nearest Neighbors regressor — returns average of K neighbors."""

    def __init__(self, k: int = 5, weighted: bool = False):
        self.k = k
        self.weighted = weighted
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y

    def predict_one(self, query: np.ndarray) -> float:
        distances = np.sqrt(np.sum((self.X_train - query) ** 2, axis=1))
        k_nearest_idx = np.argpartition(distances, self.k)[:self.k]

        if self.weighted:
            # Inverse-distance weighting: closer neighbors matter more
            k_distances = distances[k_nearest_idx]
            weights = 1.0 / (k_distances + 1e-10)
            return np.average(self.y_train[k_nearest_idx], weights=weights)
        else:
            return np.mean(self.y_train[k_nearest_idx])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_one(x) for x in X])
```

### Using scikit-learn

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Classification
X_train, y_train = load_data()

# IMPORTANT: scale features so distance is meaningful
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="distance")
knn.fit(X_scaled, y_train)

# Select K via cross-validation
for k in [3, 5, 7, 11, 15, 21]:
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X_scaled, y_train, cv=5, scoring="accuracy")
    print(f"K={k}: accuracy={scores.mean():.3f} ± {scores.std():.3f}")

# Regression (e.g., predict watch time)
reg = KNeighborsRegressor(n_neighbors=7, weights="distance")
reg.fit(X_scaled, y_continuous)
predicted_watch_time = reg.predict(scaler.transform(X_test))
```

---

## 4. Worked Example: Movie Recommendation

### Setup

```
Training data (6 users, 2 features):

User    avg_rating_action  avg_rating_comedy  label
─────────────────────────────────────────────────────
Alice        4.5               1.2            action_fan
Bob          4.0               1.5            action_fan
Charlie      4.8               2.0            action_fan
Diana        1.0               4.5            comedy_fan
Eve          1.5               4.0            comedy_fan
Frank        2.0               4.8            comedy_fan

Query: New user Grace has avg_rating_action=3.8, avg_rating_comedy=2.5
What is she? K=3
```

### Step-by-Step

```
Step 1: Compute Euclidean distances from Grace (3.8, 2.5) to all users

  d(Grace, Alice)   = √((3.8-4.5)² + (2.5-1.2)²) = √(0.49 + 1.69) = √2.18 = 1.48
  d(Grace, Bob)     = √((3.8-4.0)² + (2.5-1.5)²) = √(0.04 + 1.00) = √1.04 = 1.02
  d(Grace, Charlie) = √((3.8-4.8)² + (2.5-2.0)²) = √(1.00 + 0.25) = √1.25 = 1.12
  d(Grace, Diana)   = √((3.8-1.0)² + (2.5-4.5)²) = √(7.84 + 4.00) = √11.84 = 3.44
  d(Grace, Eve)     = √((3.8-1.5)² + (2.5-4.0)²) = √(5.29 + 2.25) = √7.54 = 2.75
  d(Grace, Frank)   = √((3.8-2.0)² + (2.5-4.8)²) = √(3.24 + 5.29) = √8.53 = 2.92

Step 2: Sort by distance, pick K=3 nearest
  1. Bob        (1.02) → action_fan
  2. Charlie    (1.12) → action_fan
  3. Alice      (1.48) → action_fan

Step 3: Majority vote
  action_fan: 3 votes, comedy_fan: 0 votes

Prediction: Grace is an action_fan ✓
```

---

## 5. Complexity Analysis

| Operation | Brute-Force KNN | KD-Tree | Ball Tree |
|-----------|----------------|---------|-----------|
| **Build** | O(1) — just store data | O(N log N) | O(N log N) |
| **Query** | O(N × D) | O(D × log N) avg, O(N × D) worst | O(D × log N) avg |
| **Memory** | O(N × D) | O(N × D) | O(N × D) |

Where N = number of training points, D = number of dimensions.

**Critical problem**: At scale (N=10M, D=128), brute-force KNN takes seconds per query. This is why we need ANN (see `ann.md`).

---

## 6. Strengths and Weaknesses

| Strengths | Weaknesses |
|-----------|------------|
| No training phase | Slow inference: O(N×D) per query |
| Naturally handles multi-class | Requires all data in memory |
| Non-parametric: adapts to any shape | Curse of dimensionality (D > 20 degrades) |
| Easy to understand and debug | Must choose K and distance metric |
| New data added instantly (no retrain) | Sensitive to irrelevant features |
| Distance-weighted version handles noise | Feature scaling is required |

---

## 7. Applied Scenarios in Recommendation Systems

### Scenario 1: User-Based Collaborative Filtering

```
"Find K users most similar to the target user → recommend what they watched"

Users represented by their rating vectors:
  user_382 = [4, 0, 5, 3, 0, 1, 0, 5, ...]  (ratings over 10K movies)

KNN with cosine similarity:
  neighbors = KNN(user_382, K=20, metric="cosine")
  recommendations = movies watched by neighbors but NOT by user_382
```

**Where used**: Early Netflix, Amazon collaborative filtering.

### Scenario 2: Item-Based Collaborative Filtering

```
"Find K items most similar to items the user liked → recommend those"

Items represented by who interacted with them:
  movie_inception = [1, 0, 1, 1, 0, 0, 1, ...]  (which users watched)

KNN with cosine similarity:
  For each movie the user liked:
    similar = KNN(movie, K=10, metric="cosine")
    candidate_pool.extend(similar)
```

**Where used**: Amazon "Customers who bought this also bought..." (item-to-item CF).

### Scenario 3: Content-Based Retrieval (Small Scale)

```
"Find K videos with the most similar content to the user's watch history"

Videos represented by content embeddings (128-d):
  video_emb = sentence_transformer("How to train a neural network")

KNN search: find 100 most similar videos to user's profile embedding
```

**Where used**: Works for small catalogs (<100K items). For millions of items, exact KNN is too slow → must switch to ANN.

### Scenario 4: Anomaly Detection in User Behavior

```
"Flag users whose behavior is very different from their K nearest neighbors"

If a user's engagement pattern is far from their neighbors:
  → possibly a bot, fraud, or compromised account
```

---

## 8. KNN vs. Learned Models

| Aspect | KNN | Learned Model (e.g., Neural Network) |
|--------|-----|-------------------------------------|
| Training time | Zero | Hours to days |
| Inference time | O(N×D) per query | O(1) per query (fixed model size) |
| Handles non-linear boundaries | Yes (naturally) | Yes (with enough capacity) |
| Feature engineering | Critical (garbage in, garbage out) | Can learn representations |
| Scales to millions of items | No (too slow) | Yes (model size is fixed) |
| Interpretable | Yes (show the neighbors) | Harder |

**Key takeaway**: KNN is the conceptual foundation for retrieval systems. In production, exact KNN is replaced by ANN (approximate nearest neighbors) for speed, and the feature space is learned by models like Two-Tower networks instead of being hand-engineered.

---

## 9. Connection to the Video Recommendation Pipeline

```
Conceptual flow:

  Hand-crafted features + exact KNN search (textbook)
       ↓ (doesn't scale)
  Learned embeddings + exact KNN search (better features, still slow)
       ↓ (still O(N) per query)
  Learned embeddings + ANN search (production-ready)
       ↓
  Two-Tower model + FAISS/ScaNN (what YouTube actually uses)
```

KNN is the **starting point** that motivates:
1. **Why we need embeddings**: Better features → better neighbors
2. **Why we need ANN**: Exact search is too slow at scale
3. **Why we need learned similarity**: Cosine distance on raw features is weak; learned embeddings capture deeper patterns
