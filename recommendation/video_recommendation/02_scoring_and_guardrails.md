# Satisfaction Scoring & Guardrail Metrics

## Part 1: How to Combine Multiple Metrics into a Satisfaction Score

### The Core Problem

You have multiple signals (click, watch time, likes, etc.) and need ONE score to rank videos. There are three common approaches in industry, from simple to advanced.

---

### Approach A: Weighted Sum of Predicted Values (Most Common in Practice)

This is what YouTube's seminal 2019 paper describes. You train **separate models** for each engagement signal, then combine their predictions at serving time.

```
Score(user, video) = w1 * P(click)
                   + w2 * E[watch_time | click]
                   + w3 * P(like | click)
                   + w4 * P(finish | click)
                   - w5 * P(dislike | click)
```

**How it works:**
- Train a click prediction model (binary classification -> probability)
- Train a watch time prediction model (regression -> expected minutes)
- Train a like prediction model (binary classification -> probability)
- Train a completion prediction model (binary classification -> probability)
- At serving time, compute each prediction and combine with weights

**How to set the weights (w1, w2, ...):**
- Start with manual tuning based on business judgment
- Then run A/B tests: tweak weights, measure long-term user retention
- Some teams use Bayesian optimization or grid search over weight space
- YouTube reportedly uses a learned combination, but manual weights are a strong baseline

**Example concrete weights (illustrative):**
```
Score = 0.1 * P(click)
      + 1.0 * E[watch_time_minutes]
      + 0.5 * P(like)
      + 0.3 * P(finish)
      - 2.0 * P(dislike)
```

Note: watch_time is in minutes (continuous), while probabilities are 0-1, so the weights naturally differ in magnitude. The negative weight on dislike penalizes predicted-bad content.

**Why this is popular:**
- Each model can be trained/updated/debugged independently
- Easy to adjust the business tradeoff by changing weights without retraining
- Interpretable: you can see WHY a video was ranked high

---

### Approach B: Multi-Task Learning with a Shared Model

Instead of separate models, train ONE model with multiple prediction heads (output layers) sharing a common backbone.

```
                   Input Features
                        |
                  [Shared Layers]
                   /    |    \     \
              P(click) E[wt] P(like) P(finish)
```

Each head has its own loss function. The final score is still a weighted combination, but the shared representation often improves each individual prediction through transfer learning.

**This is the state-of-the-art approach** at most major companies (YouTube, TikTok, Instagram). Key papers:
- MMoE (Multi-gate Mixture of Experts) -- Google 2018
- PLE (Progressive Layered Extraction) -- Tencent 2020

---

### Approach C: Direct Optimization of a Composite Label

Create a single label per training example:

```
label = 0.1 * clicked + 1.0 * watch_minutes + 0.5 * liked + 0.3 * finished
```

Then train one regression model to predict this composite label directly.

**Pros:** Simplest to implement.
**Cons:** Loses interpretability. Hard to adjust tradeoffs without retraining. Conflates different signal types. Generally NOT recommended for production systems.

---

### Recommendation for Interview

**Lead with Approach A, then upgrade to Approach B.** Show you understand both the simple baseline and the state-of-the-art. Mention Approach C only to explain why you rejected it.

---

## Part 2: How Guardrail Metrics Actually Work

Guardrail metrics are things you must NOT degrade while optimizing your primary metric. They are NOT part of the scoring formula. They operate as **separate enforcement mechanisms** at different layers of the system.

### Where Guardrails Live in the System

```
            Video Candidates (retrieval)
                    |
                    v
          +---------+---------+
          | FILTERING LAYER   |  <-- Guardrail: Hard removal
          +---------+---------+
                    |
                    v
          +---------+---------+
          | RANKING MODEL     |  <-- Primary: Satisfaction score
          +---------+---------+
                    |
                    v
          +---------+---------+
          | RE-RANKING LAYER  |  <-- Guardrail: Soft adjustments
          +---------+---------+
                    |
                    v
              Final Results
                    |
                    v
          +---------+---------+
          | A/B TEST ANALYSIS |  <-- Guardrail: Monitoring & rollback
          +---------+---------+
```

### Layer 1: Hard Filtering (Pre-Ranking)

Remove videos that violate absolute constraints. These are binary: the video either passes or is removed entirely.

| Guardrail | Implementation |
|---|---|
| Policy-violating content | A separate classifier flags harmful/illegal content; flagged videos are removed from candidate pool |
| Age-restricted content | If user is under 18, remove age-gated videos |
| Already watched | Remove videos the user has already seen (unless repeat-watch is expected) |
| Blocked creators | Remove content from creators the user has blocked |

**This happens BEFORE ranking**, so the ranking model never even sees these videos.

### Layer 2: Score Penalties / Boosts (In-Ranking)

Some guardrails modify the satisfaction score rather than removing candidates.

```
Final_Score = Satisfaction_Score * Quality_Multiplier

where Quality_Multiplier accounts for:
  - Clickbait penalty:     if P(click) is high but E[watch_time] is low -> reduce score
  - Low-quality penalty:   if video has many reports/dislikes -> reduce score
  - Creator authority:     boost videos from trusted/verified creators
```

**Example: The Clickbait Detector**
```
If P(click) > 0.8 AND E[watch_time] < 30 seconds:
    score *= 0.3   # Penalize likely clickbait
```

This is a simple but powerful guardrail that directly addresses the "engagement hacking" problem.

### Layer 3: Re-Ranking / Post-Processing (Post-Ranking)

After the ranking model produces a scored list, apply adjustments for system-level goals that a pointwise ranker can't capture.

| Guardrail | How it works |
|---|---|
| **Diversity** | Don't show 10 videos from the same creator or same topic. Use Maximal Marginal Relevance (MMR) or a sliding window to inject variety. |
| **Freshness** | Ensure at least N% of recommendations are recent uploads (< 24h old). Prevents the system from only showing proven old hits. |
| **Creator fairness** | Cap the number of impressions any single creator gets per time window. Prevent winner-take-all dynamics. |
| **Exploration** | Reserve ~5-10% of slots for "explore" candidates that the model is uncertain about. Solves the feedback loop / filter bubble problem. |

**Example: Diversity Re-Ranking**
```
final_list = []
for video in ranked_list:
    if count(video.category in final_list) < MAX_PER_CATEGORY:
        final_list.append(video)
    if len(final_list) == K:
        break
```

### Layer 4: Monitoring & Rollback (Post-Deployment)

Even after shipping, guardrail metrics are tracked in A/B experiments. If any guardrail metric degrades beyond a threshold, the experiment is automatically rolled back.

| Monitored Guardrail Metric | Threshold Example |
|---|---|
| Daily Active Users (DAU) | Must not drop > 0.5% |
| User survey satisfaction | Must not drop > 1% |
| Dislike rate | Must not increase > 5% |
| Diversity of content consumed | Gini coefficient must stay below X |
| Report rate | Must not increase > 10% |
| Session length variance | Monitor for unhealthy binge patterns |

**This is critical in interviews**: you should mention that guardrails are not just code, they are also **organizational processes** -- every model launch goes through a guardrail review before full rollout.

---

## Summary: How It All Fits Together

```
1. SCORING (what to optimize):
   Score = w1*P(click) + w2*E[watch_time] + w3*P(like) - w4*P(dislike)
   -> This is your PRIMARY objective. Trained via ML models.

2. GUARDRAILS (what NOT to break):
   -> Hard filters:    Remove policy-violating, age-restricted, already-seen
   -> Score penalties:  Penalize clickbait, low-quality content
   -> Re-ranking rules: Enforce diversity, freshness, fairness, exploration
   -> A/B monitoring:   Auto-rollback if DAU, satisfaction, or safety metrics degrade

Key insight: Guardrails are NOT in the loss function.
             They are separate enforcement layers around the model.
```

---

## Interview Talking Points

1. **"Why not put guardrails into the training objective?"** -- Because guardrails are often non-differentiable constraints (diversity, hard content policies) or organizational rules that change frequently. Keeping them separate allows fast iteration without model retraining.

2. **"What if your engagement score conflicts with a guardrail?"** -- The guardrail wins. That's the whole point. A video with a high engagement score but a policy violation is still removed. This is a design choice, not a bug.

3. **"How do you balance exploration vs. exploitation?"** -- Reserve a small percentage of recommendation slots for exploration (epsilon-greedy, Thompson sampling, or a dedicated explore model). This is both a guardrail (prevents filter bubbles) and a data collection strategy (gets labels for new content).
