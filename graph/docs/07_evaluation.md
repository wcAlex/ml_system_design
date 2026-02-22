# 07 — Evaluation

## Overview

Evaluation has three layers:
1. **Offline metrics** — fast iteration during model development
2. **Online A/B testing** — measure real-world impact before full launch
3. **Guardrail metrics** — hard limits that cancel a launch if violated

---

## Offline Evaluation

### Binary Classification Metrics (for pointwise ranking)

| Metric | Formula | Interpretation |
|---|---|---|
| AUC-ROC | Area under ROC curve | Overall ranking quality; threshold-independent |
| AUC-PR | Area under precision-recall curve | Better for imbalanced labels (rare positives) |
| Log Loss | -Σ y·log(p) + (1-y)·log(1-p) | Calibration — is the score a real probability? |

**Note on class imbalance:** In PYMK, positive examples (accepted connections) are far rarer than negatives. AUC-PR is more informative than AUC-ROC in this regime. A model that predicts 0 for everything gets AUC-ROC = 0.5 but AUC-PR = very low (reflecting that it never finds positives).

### Ranking Metrics (for the ranked list output)

Let K = number of slots shown to the user (e.g., K=20).

**Precision@K:**
```
P@K = (number of relevant items in top-K) / K
```
"Of the 20 people I showed, how many did the user connect with?"

**Recall@K:**
```
R@K = (number of relevant items in top-K) / (total relevant items)
```
"Of all the people the user would have connected with, how many appeared in the top 20?"

**NDCG@K (Normalized Discounted Cumulative Gain):**
```
DCG@K  = Σ (relevance_i / log2(i+1))     for i in 1..K
NDCG@K = DCG@K / IDCG@K
```
Position-aware: appearing in position 1 is much better than position 20. NDCG@K = 1.0 is a perfect ranking.

**Hit Rate@K:**
```
HR@K = 1 if any positive in top-K, else 0  (averaged over users)
```
Simplest ranking metric: "Did we show at least one good recommendation?"

### Recommended Offline Dashboard

| Metric | Primary? | Target direction |
|---|---|---|
| AUC-ROC | Yes | ↑ |
| AUC-PR | Yes | ↑ |
| NDCG@20 | Yes | ↑ |
| Precision@20 | Yes | ↑ |
| Recall@100 | Yes | ↑ |
| Log Loss | Secondary | ↓ |
| HR@5 | Secondary | ↑ |

---

## Evaluation by User Segment

Never evaluate only on aggregate metrics. Report separately by:

| Segment | Why it matters |
|---|---|
| Cold-start users (0–10 connections) | Hardest to serve; average metrics hide poor cold-start performance |
| Power users (>5,000 connections) | May inflate metrics — they connect with almost everyone shown |
| New accounts (<30 days old) | Important for onboarding funnel |
| Geography | Some regions have sparser graphs; metrics differ |
| Industry vertical | Professional networks vary in density (tech vs. manufacturing) |

A model can have great aggregate AUC but terrible performance for new users — the most important segment for network growth.

---

## Candidate Generation Evaluation

Stage 1 is evaluated separately from Stage 2 (ranking). The key metric is **Recall@K**:

```
Recall@K = (ground truth connections in top-K candidates) / (all ground truth connections)
```

If Stage 1 Recall@1000 = 60%, it means 40% of eventual connections were never even considered for ranking. No matter how good the ranker is, it can't recover those.

**Target:** Stage 1 Recall@1000 ≥ 80%

| Candidate generation method | Recall@1000 (typical) |
|---|---|
| 2nd-degree only | 50–70% |
| ANN on GNN embeddings | 60–75% |
| Hybrid (union) | 75–90% |

---

## Online A/B Testing

### Experimental Design

```
Control group:  current production PYMK (existing model)
Treatment group: new model (e.g., GNN two-tower)

Traffic split: 5% → 10% → 50% (ramping)
Minimum runtime: 2 weeks (capture weekly usage patterns)
Statistical test: two-sample z-test (proportions), or t-test for continuous metrics
Significance level: p < 0.05, power ≥ 0.8
```

### Primary Online Metrics

| Metric | Formula | Target |
|---|---|---|
| Connection Accept Rate | accepted / impressions | ↑ vs. control |
| 7-day New Connection Rate | new connections / active user | ↑ vs. control |
| PYMK Widget CTR | clicks / widget impressions | ↑ vs. control |
| Session-Level Engagement | Did the recommendation lead to profile visit? | ↑ |

### Guardrail Metrics (must not regress)

| Metric | Threshold | Action if violated |
|---|---|---|
| "Not Interested" dismiss rate | No significant increase | Kill experiment immediately |
| Block / Report rate | No significant increase | Kill experiment |
| Inbox spam rate | No increase | Kill experiment |
| P99 serving latency | < 150ms | Rollback if exceeded |

### Why Guardrails Matter

A model can improve accept rate by 5% but increase dismiss rate by 8% — this is a net-negative experience. Users who get too many irrelevant suggestions will disable the feature or churn. Guardrail metrics prevent optimizing for short-term gains at the cost of long-term trust.

---

## Calibration

Beyond ranking quality, check if the model's scores are calibrated probabilities:

```
Ideal: P(score = 0.7) → 70% of users with this score accept the connection
```

Calibration is important because:
1. Downstream A/B testing uses scores to filter candidates
2. Reason codes ("You have an 80% match") need to be meaningful
3. Score drift indicates distribution shift (model is stale)

**Calibration check:**
```python
from sklearn.calibration import calibration_curve
fraction_positive, mean_predicted = calibration_curve(y_true, y_score, n_bins=10)
# Plot fraction_positive vs. mean_predicted — should be near the diagonal
```

**Fix if miscalibrated:** Platt scaling (logistic regression on outputs) or isotonic regression.

---

## Offline → Online Correlation

A known challenge: offline AUC improvements don't always translate to online improvements. Track the correlation between your offline metrics and online A/B results over time.

If offline AUC improved by 2% but online accept rate didn't change, it likely means:
1. The offline metric doesn't capture what users actually care about
2. The model is overfitting to label noise in the offline dataset
3. There's a distribution shift between offline evaluation and serving traffic

---

## Interview Checkpoint

**Q: How do you measure the quality of Stage 1 (candidate generation)?**

Independently of the ranker. You collect a set of (user, eventually-connected-person) pairs from the future. Then you check: for each user, was the person they connected with in the top-K candidates produced by Stage 1? This is Recall@K. Precision is less important here — it's fine to have noisy candidates in Stage 1 as long as the Recall is high, because Stage 2 (ranking) will filter them.

**Q: What is the difference between AUC-ROC and NDCG?**

AUC-ROC measures binary classification quality — can you separate positives from negatives? It's threshold-independent.

NDCG measures ranking quality for a list of results. It's position-aware: a relevant item at rank 1 is worth much more than the same item at rank 20. NDCG is more appropriate for evaluating the final PYMK list because the user sees a ranked list, not just a binary classification.

**Q: How long should an A/B test run?**

At minimum 2 weeks, to capture:
- Day-of-week effects (Monday vs. weekend behavior is very different)
- Novelty effect decay (users may click more initially just because something looks new)
- Sufficient statistical power for small effect sizes

LinkedIn typically runs experiments for 2–4 weeks before making a decision.
