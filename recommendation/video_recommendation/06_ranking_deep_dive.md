# Ranking: From Basics to Production

A ground-up explanation of how ranking works in recommendation systems, building from simple sorting all the way to YouTube/Netflix-grade multi-task deep learning models.

---

## 1. What Is Ranking and Why Do We Need It?

After retrieval pulls ~300 candidate videos from multiple sources, we have a problem: **which ones go on top?** Retrieval scores from different sources aren't comparable — a 0.9 from Two-Tower doesn't mean the same thing as a 0.9 from the popularity cache.

Ranking is a **separate model** that scores every candidate on a unified scale by predicting how the user will engage with each video.

```
Retrieval says: "These 300 videos are RELEVANT to you"
Ranking says:   "HERE is the ORDER you'd most enjoy them in"
```

**Real-world analogy**: Retrieval is like a librarian pulling 300 books off the shelf that match your interests. Ranking is like a friend who knows you deeply, sorting those 300 books from "you'll love this" to "meh."

---

## 2. Building Up from Simple to Complex

### Level 1: Single Score Sorting (Simplest Baseline)

The most naive ranker: sort by a single signal.

```
Sort by popularity (view count):
  #1  "Baby Shark" (14B views)
  #2  "Despacito" (8B views)
  #3  "Shape of You" (6B views)
  ...

Problem: Same ranking for EVERY user. Not personalized at all.
```

**Where this works**: Cold-start users with zero history. This is our "popularity source" fallback.

### Level 2: Pointwise Scoring (Single Prediction)

Train a model to predict ONE engagement signal, like click-through rate (CTR).

```
For each (user, video) pair:
  features → model → P(click)

Sort by P(click) descending:
  #1  Video A: P(click) = 0.85
  #2  Video B: P(click) = 0.72
  #3  Video C: P(click) = 0.68
```

**Model options at this level**:
- Logistic regression (simple, fast, interpretable)
- Gradient boosted trees (XGBoost/LightGBM — strong baseline)
- Small neural network (2-3 layers)

**Problem**: Optimizing only for clicks leads to **clickbait**. A video with a sensational thumbnail gets clicked but abandoned after 5 seconds. The user is disappointed, but the model thinks it did a great job.

```
Video: "You Won't BELIEVE What Happens Next!!!"
  P(click) = 0.95     ← model ranks this #1
  Actual watch time: 4 seconds
  User satisfaction: very low

This is why YouTube MOVED AWAY from click prediction in 2012.
```

### Level 3: Predicting a Better Signal (Watch Time)

YouTube's pivotal 2012 shift: **stop optimizing for clicks, start optimizing for watch time**.

```
For each (user, video) pair:
  features → model → E[watch_time]

Sort by expected watch time descending:
  #1  Video A: E[watch_time] = 12.5 minutes
  #2  Video B: E[watch_time] = 8.2 minutes
  #3  Video C: E[watch_time] = 5.1 minutes
```

This was a major improvement — clickbait gets low watch time, so it naturally drops in rank.

**Problem**: Watch time alone favors LONG videos. A 3-hour documentary that users watch 20% of (36 min) outranks a 5-minute tutorial that users watch 100% of (5 min) — even though the tutorial is a better recommendation.

### Level 4: Multiple Predictions Combined (Multi-Objective)

The modern approach: predict **multiple engagement signals** and combine them.

```
For each (user, video) pair, predict ALL of these:
  P(click)      = 0.72      Will the user click?
  E[watch_time] = 8.5 min   How long will they watch?
  P(like)       = 0.18      Will they like it?
  P(finish)     = 0.45      Will they finish it?
  P(dislike)    = 0.02      Will they dislike it?

Combined score = weighted sum:
  score = 0.1 × P(click)
        + 1.0 × E[watch_time]
        + 0.5 × P(like)
        + 0.3 × P(finish)
        - 2.0 × P(dislike)

  = 0.072 + 8.5 + 0.09 + 0.135 - 0.04
  = 8.757
```

**Why this is powerful**: The weights encode business priorities. If leadership says "we care more about user satisfaction than raw engagement," increase the like weight and dislike penalty. No model retraining needed — just change the weights.

**This is what YouTube, TikTok, Instagram, and Netflix use today.**

---

## 3. The YouTube Ranking System (Real-World Case Study)

YouTube has published two landmark papers that define modern recommendation ranking.

### YouTube 2016 Paper: "Deep Neural Networks for YouTube Recommendations"

The paper that introduced deep learning to large-scale recommendation ranking.

```
Architecture:

  ┌──────────────────────────────────────────────┐
  │               Input Features                  │
  │                                                │
  │  Watch history  Search history  Demographics   │
  │  (embedded &    (embedded &     (geo, age,     │
  │   averaged)      averaged)      device, etc.)  │
  └──────────────────────┬─────────────────────────┘
                         │
                    Concatenate
                         │
                  ┌──────┴──────┐
                  │  Dense 1024  │
                  │  ReLU        │
                  ├─────────────┤
                  │  Dense 512   │
                  │  ReLU        │
                  ├─────────────┤
                  │  Dense 256   │
                  │  ReLU        │
                  └──────┬──────┘
                         │
                  Weighted logistic regression
                         │
                  E[watch_time]
```

**Key insights from the paper**:

1. **Watch time as the training label, not clicks**: YouTube trained on watch time using weighted logistic regression where positive examples are weighted by watch duration.

2. **"Example age" feature**: They added how old the training example was as a feature. At serving time, set it to 0 (present). This lets the model learn freshness preference from data rather than hard-coding it.

3. **Features matter more than model architecture**: They found that adding more features (search history, watch history demographics) improved quality more than making the model deeper.

### YouTube 2019 Paper: "Recommending What Video to Watch Next"

This paper introduced the **multi-task ranking** approach used today.

```
Key innovations:

1. Multi-task predictions:
   - Engagement tasks: click, watch time, like, share
   - Satisfaction tasks: like, dismissal (negative)
   - Combined into a single score

2. MMoE (Multi-gate Mixture of Experts):
   - Multiple expert sub-networks, each specializing in different patterns
   - Per-task gating networks that learn which experts are useful for each task
   - Solves the "task interference" problem

3. Shallow tower for position bias:
   - Users click position 1 more than position 5 regardless of content
   - A separate small network learns this position bias
   - At serving time, the position bias is REMOVED → unbiased predictions
```

```
YouTube 2019 architecture (simplified):

  Input features
       │
  ┌────┴────────────────────────────┐
  │     Expert 1   Expert 2   ...    │   ← Shared expert networks
  └────┬────────────────────────────┘
       │
  ┌────┼──────┬──────┬──────┐
  │    │      │      │      │
 Gate Gate  Gate  Gate   Gate         ← Per-task gating
  │    │      │      │      │
 Tower Tower Tower Tower  Tower       ← Task-specific heads
  │    │      │      │      │
P(click) E[wt] P(like) P(share) P(dismiss)

                    MINUS

            ┌─────────────┐
            │Shallow Tower │  ← Position bias
            │(removed at   │     (learned during training,
            │ serving)     │      subtracted at inference)
            └─────────────┘
```

---

## 4. The Netflix Ranking System

Netflix's ranking problem is different from YouTube's: instead of a single feed, Netflix shows **rows of videos** organized by genre/theme. But the core ranking ideas are the same.

### Netflix's Approach

```
Netflix's key insight: Predict PROBABILITY OF PLAY.

For each (user, video) pair:
  P(play) = probability the user will click play and watch

This is simpler than YouTube's multi-task approach because Netflix's
primary action is binary: play or don't play.

But Netflix also considers:
  - Expected hours of viewing after play
  - Probability of completing the series
  - Probability of rating highly
```

### Netflix's Personalized Ranking Layers

```
Layer 1: Row selection
  Which rows to show? ("Because you watched...", "Trending", "New Releases")
  Each row type has a different ranking model

Layer 2: Within-row ranking
  Within "Action Movies" row, which titles go first?
  Pointwise model: P(play | user, title, context)

Layer 3: Artwork personalization
  Netflix shows DIFFERENT thumbnails for the SAME movie to different users
  A romantic comedy might show the romantic scene to romance fans
  and the funny scene to comedy fans
  → This increases click rate significantly
```

### What Netflix Taught the Industry

1. **Context matters**: Same user at 8am (quick watch before work) vs 9pm (movie night) should see different rankings
2. **Artwork is part of ranking**: The thumbnail IS the recommendation — personalizing it is a huge lever
3. **Multi-armed bandits for exploration**: Netflix uses Thompson Sampling to balance showing known-good titles vs exploring new ones

---

## 5. Pointwise vs. Pairwise vs. Listwise Ranking

Three paradigms for training ranking models. This matters for interviews.

### Pointwise (Most Common in Production)

Treat each (user, item) pair independently. Predict an absolute score.

```
Training examples:
  (user_1, video_A) → P(click) = 1  (clicked)
  (user_1, video_B) → P(click) = 0  (not clicked)
  (user_1, video_C) → P(click) = 1  (clicked)

Loss: Binary cross-entropy (per example)
  L = -[y·log(p) + (1-y)·log(1-p)]

Sort by predicted scores at serving time.
```

**Pros**: Simple, scalable, each example is independent (easy to parallelize).
**Cons**: Doesn't directly optimize for ranking quality — treats each prediction in isolation.

**Used at**: YouTube, TikTok, most production systems.

### Pairwise (Better Ranking Quality)

Train on **pairs** of items: "Video A should rank above Video B."

```
Training examples (pairs):
  (user_1, video_A > video_B)  → A was clicked, B wasn't
  (user_1, video_A > video_C)  → A had longer watch time

Loss: Pairwise logistic loss
  L = log(1 + exp(score_B - score_A))
  Penalizes when B is scored higher than A (wrong order)

Key algorithms:
  - RankNet (Microsoft, 2005)
  - LambdaMART (used in web search ranking)
```

**Pros**: Directly optimizes relative ordering (what we actually care about).
**Cons**: O(N²) pairs per user — expensive. Hard to scale.

**Used at**: Bing search ranking, some e-commerce systems.

### Listwise (Theoretically Best)

Optimize the **entire ranked list** as a whole. Directly optimize NDCG or other list-level metrics.

```
Training: Given all items for a user, optimize the permutation.
  Directly maximizes NDCG, MAP, or other ranking metrics.

Key algorithms:
  - ListNet
  - SoftRank
  - ApproxNDCG (approximate NDCG for gradient-based optimization)
```

**Pros**: Directly optimizes the metric we evaluate on.
**Cons**: Computationally expensive, hard to scale, marginal improvement over pointwise in practice.

**Used at**: Research, some search engines. Rarely in production recommendations.

### Which to Use?

```
Interview answer:

"In production recommendation ranking, pointwise is dominant because:
  1. Scale: Billions of training examples are independent → easy distributed training
  2. Multi-task: Each task has its own loss (BCE for click, Huber for watch time)
  3. Calibration: Pointwise models predict actual probabilities (P(click)=0.05 means 5%)
     which is needed for combining predictions with fixed weights
  4. The ranking quality gap vs. pairwise is small and not worth the complexity

 Pairwise is more common in search ranking (Bing, Google) where the query-document
 relevance signal is clearer."
```

---

## 6. The Multi-Task Ranking Problem Explained

### Why Multiple Tasks?

A single prediction is a lossy summary of user engagement. Different signals capture different aspects:

```
Signal        What it captures                When it misleads
─────────────────────────────────────────────────────────────────
P(click)      User curiosity / interest       Clickbait: high click, low value
E[watch_time] Engagement depth                Favors long videos regardless of quality
P(like)       User satisfaction               Sparse signal (few users like)
P(finish)     Content quality / completion    Short videos have unfair advantage
P(dislike)    Negative signal                 Very sparse, but very important
```

No single signal tells the full story. The combination tells a much richer story.

### The Task Interference Problem

When you train multiple tasks in one model, they can **hurt each other**:

```
Shared-bottom architecture (naive multi-task):

  features → [Shared MLP] → Task 1 head → P(click)
                           → Task 2 head → E[watch_time]

Problem: The shared layers must serve BOTH tasks.
  Click prediction wants: features about thumbnail attractiveness, title curiosity
  Watch time wants: features about content quality, topic match, video length

  These can CONFLICT: a clickbaity thumbnail helps predict clicks
  but hurts watch time prediction. The shared layers are pulled in both directions.
  → Both tasks get worse than if trained separately.
```

### How MMoE Solves This

**Multi-gate Mixture of Experts** (Google, 2018) is the standard solution.

```
Instead of one shared layer, create MULTIPLE expert sub-networks.
Each task has its own GATE that learns which experts are useful for it.

  features
     │
  ┌──┼──┬──┬──┐
  │  │  │  │  │
  E₁ E₂ E₃ E₄ E₅    ← 5 expert networks (each is a 2-layer MLP)
  │  │  │  │  │       Each expert can specialize in different patterns
  └──┼──┴──┴──┘
     │
  ┌──┴──┐
  │     │
 G₁    G₂               ← Per-task gates (softmax over expert outputs)
  │     │
 T₁    T₂               ← Task-specific towers
  │     │
P(click)  E[watch_time]

Gate 1 (click task) might learn:
  "Use 60% Expert 1 + 30% Expert 3 + 10% Expert 5"
  Expert 1 specializes in visual/thumbnail features
  Expert 3 specializes in title/curiosity features

Gate 2 (watch time task) might learn:
  "Use 50% Expert 2 + 40% Expert 4 + 10% Expert 1"
  Expert 2 specializes in content quality features
  Expert 4 specializes in user-content match features

Result: Each task gets a CUSTOMIZED representation from the experts.
No more conflict — click and watch time tasks use different expert mixes.
```

### Concrete Example: How MMoE Helps

```
Video: "SHOCKING: Celebrity Does Something WILD" (clickbait)

Expert 1 (visual patterns):   "Bright thumbnail, all-caps title → high curiosity"
Expert 2 (content quality):   "Low completion rate, high dislike ratio → poor content"
Expert 3 (topic match):       "Celebrity gossip, user watches tech → weak match"

Gate for P(click):
  Weights: E1=0.7, E2=0.1, E3=0.2
  → Heavily uses visual patterns → predicts P(click) = 0.75 (high)

Gate for E[watch_time]:
  Weights: E1=0.1, E2=0.6, E3=0.3
  → Heavily uses content quality → predicts E[watch_time] = 15 seconds (low)

Combined score: 0.1 × 0.75 + 1.0 × 0.25 + ... - 2.0 × P(dislike)
  → LOW score despite high click probability
  → Clickbait naturally demoted
```

---

## 7. Feature Importance in Ranking

From YouTube's and Netflix's experience, features matter more than model architecture. Here's a priority ordering:

### Tier 1: Highest Impact Features

```
1. User-item interaction history
   "Has this user watched videos from this creator before?"
   "How much of this category does the user watch?"
   → These are the STRONGEST signals in any ranking model

2. Item engagement statistics
   "What is this video's historical CTR?"
   "What is the average completion rate?"
   → Global quality signal

3. Collaborative filtering embeddings
   "User embedding · Item embedding from Two-Tower model"
   → Captures deep user-item affinity
```

### Tier 2: Important Context

```
4. Time context
   "What time of day is it?" "Weekend or weekday?"
   → People watch different content at different times

5. Device context
   "Mobile or TV?" "On WiFi or cellular?"
   → Mobile users prefer shorter videos

6. Session context
   "How many videos has the user watched this session?"
   "What was the last video they watched?"
   → Session fatigue and topic momentum
```

### Tier 3: Helpful Refinement

```
7. Video freshness
   "How old is this video?"
   → Fresh content gets a boost

8. Creator features
   "How many subscribers does this creator have?"
   "What's the creator's average video quality?"

9. Content features
   "Video duration", "Language", "Has subtitles?"
```

---

## 8. Practical Challenges in Production Ranking

### Challenge 1: Selection Bias

```
Problem:
  The model only sees data about items that were SHOWN to users.
  Items that were never shown have no training signal.
  → The model learns to reinforce its own past decisions.

Example:
  Model ranks Video X at position 50 → user never scrolls that far → no click
  Model learns: "Video X gets no clicks" → ranks it even lower
  But Video X might be great — the user just never saw it.

YouTube's solution:
  - Log impressions with position information
  - Use position as a feature during training, remove at serving
  - Inverse propensity weighting: upweight rare position impressions
  - Reserve 5-10% of traffic for exploration (random ranking)
```

### Challenge 2: Position Bias

```
Problem:
  Users click position 1 more than position 5, regardless of quality.
  Naive model: "Position 1 items always get clicked → they must be better"
  Reality: Users click position 1 because it's at the top.

                  Actual CTR by position (all else equal):
                  Position 1: 25%
                  Position 2: 15%
                  Position 3: 10%
                  Position 5:  5%
                  Position 10: 2%

YouTube 2019 solution — Shallow Tower:
  1. Train a small "shallow tower" network that takes ONLY position as input
  2. It learns the position bias: P(click | position)
  3. The main model learns: P(click | user, video, context, position)
  4. At serving time: subtract the shallow tower's output
     → gives P(click | user, video, context) without position contamination
```

### Challenge 3: Delayed Feedback

```
Problem:
  When the model scores a video, it predicts P(like).
  But the user might like the video 3 hours later (after finishing it).
  If we train on data collected immediately after impression:
    → Many "no like" labels are actually "hasn't liked YET"

Solutions:
  - Wait for an attribution window (e.g., 24 hours) before creating training labels
  - Update labels retroactively when delayed actions arrive
  - Use shorter-term signals (watch time) as proxies for longer-term signals (like)
```

### Challenge 4: Training-Serving Skew

```
Problem:
  Features computed during training (offline, Spark) may differ
  slightly from features computed at serving (online, Redis).

  Example:
    Training: user_avg_watch_time computed from daily Spark job
              → uses data up to midnight
    Serving:  user_avg_watch_time computed from streaming pipeline
              → includes today's watches

  If the values differ → model predictions are unreliable

Netflix's solution:
  - Log the EXACT features used at serving time (feature logging)
  - Use those logged features for training (instead of recomputing)
  - Periodically compare training features vs serving features
  - Alert if distribution drift exceeds threshold
```

---

## 9. Score Combination: How Weights Are Tuned

The final ranking score is:

```
score = w₁ × P(click) + w₂ × E[watch_time] + w₃ × P(like) + w₄ × P(finish) - w₅ × P(dislike)
```

**How do you choose the weights?**

### Approach 1: Manual Tuning with A/B Tests

```
1. Start with reasonable defaults:
   w = {click: 0.1, watch_time: 1.0, like: 0.5, finish: 0.3, dislike: -2.0}

2. Hypothesis: "We should weight likes more"
   Treatment: {click: 0.1, watch_time: 1.0, like: 1.0, finish: 0.3, dislike: -2.0}

3. Run A/B test for 2 weeks, measure:
   - Total watch time per user per day (primary metric)
   - Like rate, dislike rate, DAU retention (guardrails)

4. If treatment wins on primary metric without violating guardrails → ship it
```

### Approach 2: Grid Search / Bayesian Optimization

```
1. Define a grid of weight combinations
2. For each combination, run offline simulation:
   - Replay historical data with those weights
   - Measure simulated NDCG, diversity, watch time
3. Pick top-3 candidates from offline
4. A/B test the top candidates online
```

### Approach 3: Learned Combination (Advanced)

```
Instead of fixed weights, train a SECOND model that learns to combine predictions:

  combined_score = MLP(P(click), E[watch_time], P(like), P(finish), P(dislike))

Training label: actual user satisfaction (e.g., did user return next day?)

Pros: Can learn non-linear combinations (e.g., "high click + low watch = penalty")
Cons: Another model to maintain, harder to interpret
```

---

## 10. Evolution of Ranking at YouTube: A Timeline

```
2008: Sort by view count
      → Everyone sees the same ranking
      → Heavily favors old popular videos

2012: Predict watch time (logistic regression)
      → Personalized, reduces clickbait
      → YouTube's most impactful change

2016: Deep neural network (DNN)
      → Published "Deep Neural Networks for YouTube Recommendations"
      → Rich features: watch history, search history, demographics
      → Major quality improvement

2018: Multi-gate Mixture of Experts (MMoE)
      → Google publishes MMoE paper
      → Multiple tasks without interference
      → Adopted across YouTube, Google Ads

2019: Multi-task ranking with shallow tower
      → Published "Recommending What Video to Watch Next"
      → Position bias correction via shallow tower
      → Separate engagement and satisfaction tasks

2020+: Increasing focus on satisfaction over engagement
      → "Responsible recommendations" initiative
      → Down-rank borderline content even if it gets clicks
      → Long-term user satisfaction metrics (7-day retention)
```

---

## 11. Interview Cheat Sheet: Ranking

### "Walk me through how you'd design a ranking model"

> **Step 1**: Define what to predict. I'd start with multiple objectives — click, watch time, like, and dislike — because no single signal captures user satisfaction fully.
>
> **Step 2**: Choose the architecture. MMoE (Multi-gate Mixture of Experts) is the industry standard for multi-task ranking. It uses shared expert networks with per-task gating to avoid task interference.
>
> **Step 3**: Feature engineering. The most impactful features are user-item interaction history, item engagement stats, and collaborative filtering embeddings. Context features (time, device, session) add meaningful lift.
>
> **Step 4**: Handle biases. Position bias via a shallow tower (YouTube 2019 approach). Selection bias via exploration and inverse propensity weighting.
>
> **Step 5**: Score combination. Weighted sum of task predictions with weights tuned via A/B testing. The weights encode business priorities.
>
> **Step 6**: Evaluation. Offline: per-task AUC, NDCG. Online: A/B test measuring total watch time, like rate, and DAU retention as guardrails.

### Common follow-up questions

| Question | Key Point |
|----------|-----------|
| "Why not just predict watch time?" | Favors long videos, misses satisfaction signals like likes |
| "Why MMoE over shared-bottom?" | Shared-bottom causes task interference; MMoE's per-task gating avoids this |
| "How do you handle clickbait?" | Multi-task: high P(click) + low E[watch_time] = low combined score |
| "How do you tune the combination weights?" | A/B testing different weight configs; no single offline metric determines them |
| "What's the biggest practical challenge?" | Training-serving skew — feature values at serving time must match training time |
| "Pointwise vs. pairwise for ranking?" | Pointwise dominates in practice: simpler, scalable, calibrated probabilities for score combination |

---

## 12. Relationship to Other Components

```
For implementation details (architecture, code, features):
  → models/03_multi_task_ranking/README.md

For the serving infrastructure that hosts this model:
  → 05_serving_system.md (Section 2.5: Model Serving)

For what happens AFTER ranking:
  → 07_reranking_deep_dive.md
```
