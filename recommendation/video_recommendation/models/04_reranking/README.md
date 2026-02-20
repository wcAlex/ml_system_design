# Re-Ranking Module (Post-Processing & Guardrails)

## 1. Role in the Pipeline

The Re-Ranking module is **Stage 3** of the recommendation pipeline. It takes the top ~50-100 scored candidates from the ranking model and applies **business rules, diversity constraints, safety filters, and exploration** to produce the final 20-50 recommendations shown to the user.

**Why a separate re-ranking stage?**
- Business rules (diversity, freshness, creator fairness) are **non-differentiable** — hard to bake into training loss
- Safety filters are **binary hard constraints** (must block, not just down-rank)
- Rules can be **updated without retraining** any ML model
- Enables **rapid policy iteration** (new content policy → deploy in hours, not days)

```
Ranked candidates (top 100, sorted by ranking_score)
         │
    ┌────┴────┐
    │ FILTER  │  ← Hard removal: already watched, policy violations, age-restricted
    └────┬────┘
         │
    ┌────┴────┐
    │ ADJUST  │  ← Score penalties/boosts: clickbait, quality, freshness
    └────┬────┘
         │
    ┌────┴────┐
    │ REORDER │  ← Diversity, creator caps, exploration slots
    └────┬────┘
         │
    Final 20-50 recommendations
```

---

## 2. Components

### Component A: Hard Filters (Pre-Ranking)

These filters run **before or at the top** of re-ranking. Any candidate that fails is removed entirely.

| Filter | Logic | Source |
|--------|-------|--------|
| Already watched | `video_id in user_watch_history` | User profile store |
| Policy violation | `video.policy_status == "blocked"` | Trust & Safety DB |
| Age restriction | `video.age_restricted AND user.age < 18` | Video metadata + user profile |
| Blocked creator | `video.creator_id in user.blocked_creators` | User settings |
| Region restriction | `user.country not in video.allowed_regions` | Video metadata |
| Copyright claim | `video.copyright_status == "claimed"` | Content ID system |

### Component B: Score Adjustments (Soft Penalties/Boosts)

These modify the ranking score without removing candidates.

#### Clickbait Detector

```python
def clickbait_penalty(predictions: dict) -> float:
    """
    Detect and penalize clickbait: high click probability but low engagement.

    Clickbait signal: Users click (attracted by thumbnail/title) but
    quickly abandon (poor content quality).
    """
    p_click = predictions["click"]
    e_watch_time = predictions["watch_time"]
    p_finish = predictions["finish"]

    # High click + low watch = clickbait signal
    if p_click > 0.7 and e_watch_time < 30 and p_finish < 0.2:
        return 0.3  # 30% penalty
    elif p_click > 0.6 and e_watch_time < 60 and p_finish < 0.3:
        return 0.15  # 15% penalty
    return 0.0
```

#### Quality Score

```python
def quality_multiplier(video_stats: dict) -> float:
    """
    Boost high-quality content, penalize low-quality.

    Quality signals come from historical engagement data.
    """
    like_ratio = video_stats["like_ratio"]       # likes / (likes + dislikes)
    report_rate = video_stats["report_rate"]      # reports / impressions
    avg_completion = video_stats["avg_completion"] # avg watch %

    quality = 1.0

    # High completion + high like ratio = quality content
    if avg_completion > 0.7 and like_ratio > 0.95:
        quality = 1.15  # 15% boost

    # High report rate = low quality
    if report_rate > 0.001:  # 0.1% of viewers report
        quality = 0.7  # 30% penalty

    # Very low like ratio
    if like_ratio < 0.8:
        quality *= 0.85

    return quality
```

#### Freshness Boost

```python
def freshness_boost(video_upload_age_hours: float) -> float:
    """
    Boost recently uploaded videos to ensure new content gets exposure.

    Critical for creator ecosystem health: if new uploads never get shown,
    creators stop uploading.
    """
    if video_upload_age_hours < 1:
        return 1.3    # 30% boost for very fresh content
    elif video_upload_age_hours < 6:
        return 1.15   # 15% boost
    elif video_upload_age_hours < 24:
        return 1.05   # 5% boost
    return 1.0
```

### Component C: Diversity Re-Ordering

The most algorithmically interesting part of re-ranking.

#### Option 1: Maximal Marginal Relevance (MMR)

**Greedily selects items that are both relevant and diverse.**

```
Score_MMR(v) = λ * relevance(v) - (1-λ) * max_similarity(v, already_selected)
```

Where λ controls the relevance-diversity trade-off.

#### Option 2: Sliding Window Constraints

Apply hard constraints over windows of the recommendation list:
- Max 2 videos from same creator in any window of 10
- Max 3 videos from same category in any window of 10
- At least 1 video < 24h old in every window of 5

#### Option 3: Determinantal Point Process (DPP)

Probabilistic model that naturally encourages diversity. DPP assigns higher probability to subsets where items are dissimilar.

**Trade-off**: More principled than heuristic rules but significantly harder to implement and debug.

### Comparison

| Approach | Diversity Quality | Interpretability | Implementation | Latency |
|----------|------------------|-----------------|----------------|---------|
| MMR | Good | High | Simple | Low |
| Sliding Window | Moderate | Highest | Simplest | Lowest |
| DPP | Best | Low | Complex | Medium |

**Recommendation**: Start with sliding window constraints for hard rules + MMR for soft diversity.

---

## 3. Data Requirements

Re-ranking uses data differently from ML models — it primarily needs **lookup tables and real-time signals**, not training data.

| Data | Source | Latency Requirement |
|------|--------|-------------------|
| User watch history (for dedup) | User profile store (Redis) | < 5ms |
| Video policy status | Trust & Safety DB | Pre-cached |
| Video engagement stats | Feature store | < 10ms |
| User block list | User settings DB | < 5ms |
| Video content embeddings | Embedding store (for MMR) | Pre-cached |
| Creator impression counts | Real-time counter (Redis) | < 5ms |
| A/B test config (slot allocation, weights) | Config service | Cached |

---

## 4. Full Re-Ranking Pipeline Code

```python
import numpy as np
from dataclasses import dataclass


@dataclass
class RankedCandidate:
    video_id: str
    ranking_score: float
    predictions: dict        # {click, watch_time, like, finish, dislike}
    video_metadata: dict     # {creator_id, category, upload_age_hours, ...}
    video_stats: dict        # {like_ratio, report_rate, avg_completion, ...}
    content_embedding: np.ndarray  # For diversity computation


class ReRanker:
    """
    Full re-ranking pipeline: filter → adjust → diversify → explore.
    """

    def __init__(self, config: dict = None):
        self.config = config or {
            "max_per_creator": 2,       # Max videos per creator in final list
            "max_per_category": 4,      # Max videos per category
            "exploration_fraction": 0.1, # 10% exploration slots
            "freshness_slots": 2,       # Reserved slots for fresh content
            "mmr_lambda": 0.7,          # Relevance vs diversity trade-off
            "final_list_size": 30,      # Number of final recommendations
        }

    def rerank(self, candidates: list[RankedCandidate],
               user_context: dict) -> list[RankedCandidate]:
        """
        Full re-ranking pipeline.

        Args:
            candidates: ~100 ranked candidates from Stage 2
            user_context: {user_id, watch_history, blocked_creators, age, country, ...}
        """
        # Step 1: Hard filters
        filtered = self._apply_hard_filters(candidates, user_context)

        # Step 2: Score adjustments
        adjusted = self._adjust_scores(filtered)

        # Step 3: Diversity re-ordering (MMR)
        diverse = self._mmr_diversify(adjusted)

        # Step 4: Apply slot constraints
        constrained = self._apply_slot_constraints(diverse)

        # Step 5: Insert exploration candidates
        final = self._insert_exploration(constrained, filtered)

        return final[:self.config["final_list_size"]]

    # ─── Step 1: Hard Filters ───

    def _apply_hard_filters(self, candidates, user_context):
        watch_history = set(user_context.get("watch_history", []))
        blocked_creators = set(user_context.get("blocked_creators", []))
        user_age = user_context.get("age", 99)
        user_country = user_context.get("country", "US")

        filtered = []
        for c in candidates:
            # Already watched
            if c.video_id in watch_history:
                continue
            # Blocked creator
            if c.video_metadata.get("creator_id") in blocked_creators:
                continue
            # Policy violation
            if c.video_metadata.get("policy_status") == "blocked":
                continue
            # Age restriction
            if c.video_metadata.get("age_restricted") and user_age < 18:
                continue
            # Region restriction
            allowed = c.video_metadata.get("allowed_regions")
            if allowed and user_country not in allowed:
                continue

            filtered.append(c)

        return filtered

    # ─── Step 2: Score Adjustments ───

    def _adjust_scores(self, candidates):
        for c in candidates:
            # Clickbait penalty
            penalty = clickbait_penalty(c.predictions)
            c.ranking_score *= (1 - penalty)

            # Quality multiplier
            c.ranking_score *= quality_multiplier(c.video_stats)

            # Freshness boost
            upload_age = c.video_metadata.get("upload_age_hours", 9999)
            c.ranking_score *= freshness_boost(upload_age)

        # Re-sort after adjustments
        candidates.sort(key=lambda x: x.ranking_score, reverse=True)
        return candidates

    # ─── Step 3: MMR Diversity ───

    def _mmr_diversify(self, candidates):
        """Maximal Marginal Relevance for diversity."""
        if not candidates:
            return []

        lam = self.config["mmr_lambda"]
        target_size = self.config["final_list_size"] + 10  # Over-select

        selected = [candidates[0]]
        remaining = candidates[1:]

        while remaining and len(selected) < target_size:
            best_score = -float("inf")
            best_idx = 0

            for i, cand in enumerate(remaining):
                # Relevance component (normalized)
                relevance = cand.ranking_score

                # Diversity component: max similarity to already selected
                max_sim = max(
                    np.dot(cand.content_embedding, s.content_embedding)
                    for s in selected
                )

                mmr_score = lam * relevance - (1 - lam) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    # ─── Step 4: Slot Constraints ───

    def _apply_slot_constraints(self, candidates):
        """Enforce per-creator and per-category caps."""
        creator_counts = {}
        category_counts = {}
        constrained = []

        for c in candidates:
            creator = c.video_metadata.get("creator_id", "unknown")
            category = c.video_metadata.get("category", "unknown")

            creator_count = creator_counts.get(creator, 0)
            category_count = category_counts.get(category, 0)

            if (creator_count >= self.config["max_per_creator"] or
                category_count >= self.config["max_per_category"]):
                continue

            constrained.append(c)
            creator_counts[creator] = creator_count + 1
            category_counts[category] = category_count + 1

        return constrained

    # ─── Step 5: Exploration ───

    def _insert_exploration(self, main_list, full_candidates):
        """
        Reserve slots for exploration candidates.

        Exploration helps:
        - Discover new user interests (avoid filter bubbles)
        - Collect training data for less-exposed videos
        - Improve long-term recommendation quality
        """
        n_explore = max(1, int(
            self.config["final_list_size"] * self.config["exploration_fraction"]
        ))

        main_ids = set(c.video_id for c in main_list)

        # Exploration candidates: videos NOT in main list,
        # sampled proportional to ranking score (softmax)
        explore_pool = [c for c in full_candidates if c.video_id not in main_ids]

        if not explore_pool:
            return main_list

        # Softmax sampling (not pure random — still somewhat relevant)
        scores = np.array([c.ranking_score for c in explore_pool])
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()

        explore_indices = np.random.choice(
            len(explore_pool), size=min(n_explore, len(explore_pool)),
            replace=False, p=probs
        )
        explore_candidates = [explore_pool[i] for i in explore_indices]

        # Insert exploration candidates at random positions (not all at the end)
        result = list(main_list)
        for exp_cand in explore_candidates:
            insert_pos = np.random.randint(
                len(result) // 3,  # Not in top third
                len(result)
            )
            result.insert(insert_pos, exp_cand)

        return result
```

---

## 5. Model Input / Output Examples

### Input to Re-Ranking

```python
# Top 100 candidates from ranking model (already scored)
candidates = [
    RankedCandidate(
        video_id="v_48291",
        ranking_score=7.32,
        predictions={"click": 0.85, "watch_time": 5.9, "like": 0.22,
                     "finish": 0.58, "dislike": 0.01},
        video_metadata={"creator_id": "c_100", "category": "gaming",
                       "upload_age_hours": 3, "policy_status": "ok"},
        video_stats={"like_ratio": 0.96, "report_rate": 0.0001,
                    "avg_completion": 0.72},
        content_embedding=np.array([0.03, -0.11, ...]),  # 128-d
    ),
    RankedCandidate(
        video_id="v_19283",
        ranking_score=5.09,
        predictions={"click": 0.72, "watch_time": 4.8, "like": 0.18,
                     "finish": 0.45, "dislike": 0.02},
        video_metadata={"creator_id": "c_200", "category": "education",
                       "upload_age_hours": 48, "policy_status": "ok"},
        video_stats={"like_ratio": 0.94, "report_rate": 0.0,
                    "avg_completion": 0.68},
        content_embedding=np.array([0.05, -0.08, ...]),
    ),
    # ... 98 more candidates
]

user_context = {
    "user_id": "u_382910",
    "watch_history": ["v_10001", "v_10002", ...],  # last 1000 videos
    "blocked_creators": ["c_999"],
    "age": 28,
    "country": "US",
}
```

### Output from Re-Ranking

```python
final_recommendations = [
    # Position 1: Top gaming video (fresh, high score)
    {"video_id": "v_48291", "score": 8.77, "source": "main"},

    # Position 2: Education video
    {"video_id": "v_19283", "score": 5.09, "source": "main"},

    # Position 3: Music video (different category for diversity)
    {"video_id": "v_77234", "score": 4.92, "source": "main"},

    # ... positions 4-8: various categories, different creators

    # Position 9: Exploration candidate (cooking video - new interest?)
    {"video_id": "v_91002", "score": 2.15, "source": "exploration"},

    # ... positions 10-28

    # Position 29: Fresh upload (< 1 hour old, freshness boosted)
    {"video_id": "v_99501", "score": 3.41, "source": "freshness_slot"},

    # Position 30: Another exploration candidate
    {"video_id": "v_62019", "score": 1.87, "source": "exploration"},
]

# Summary of final list:
# - 27 main candidates (diverse, creator-capped)
# - 2 freshness slots (content < 24h old)
# - 3 exploration candidates (inserted at random positions in bottom 2/3)
# - Unique creators: 24 (max 2 per creator)
# - Categories represented: 8 out of 20
```

---

## 6. Evaluation Methods

Re-ranking evaluation is different from ML model evaluation — it's primarily about **constraint satisfaction and user experience metrics**.

### Offline Metrics

| Metric | Description | How to Compute |
|--------|-------------|----------------|
| **Intra-list diversity (ILD)** | Average pairwise distance between recommended items | `mean(1 - cosine_sim(v_i, v_j))` for all pairs |
| **Coverage** | % of video catalog that gets recommended to at least one user | Count unique videos across all users |
| **Creator fairness (Gini)** | How evenly impressions are distributed across creators | Gini coefficient of creator impression counts |
| **Category entropy** | How diverse categories are in the final list | Shannon entropy of category distribution |
| **Filter effectiveness** | % of policy-violating content blocked | Must be 100% |
| **NDCG degradation** | How much relevance drops after re-ranking vs. pure ranking | Should be < 5% degradation |

### Evaluation Code

```python
def evaluate_reranking(original_ranked, reranked, user_ground_truth):
    """Evaluate the re-ranking module's effect on diversity and relevance."""

    metrics = {}

    # 1. Relevance preservation (NDCG comparison)
    ndcg_before = compute_ndcg_from_list(original_ranked, user_ground_truth)
    ndcg_after = compute_ndcg_from_list(reranked, user_ground_truth)
    metrics["ndcg_before_rerank"] = ndcg_before
    metrics["ndcg_after_rerank"] = ndcg_after
    metrics["ndcg_degradation"] = (ndcg_before - ndcg_after) / ndcg_before

    # 2. Intra-list diversity
    embeddings = [c.content_embedding for c in reranked]
    n = len(embeddings)
    total_dist = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim = np.dot(embeddings[i], embeddings[j])
            total_dist += (1 - sim)
            count += 1
    metrics["intra_list_diversity"] = total_dist / count if count > 0 else 0

    # 3. Category entropy
    categories = [c.video_metadata["category"] for c in reranked]
    from collections import Counter
    cat_counts = Counter(categories)
    total = sum(cat_counts.values())
    probs = [c / total for c in cat_counts.values()]
    metrics["category_entropy"] = -sum(p * np.log2(p) for p in probs if p > 0)

    # 4. Creator concentration
    creators = [c.video_metadata["creator_id"] for c in reranked]
    creator_counts = Counter(creators)
    metrics["unique_creators"] = len(creator_counts)
    metrics["max_creator_count"] = max(creator_counts.values())

    # 5. Freshness
    fresh = sum(1 for c in reranked
                if c.video_metadata.get("upload_age_hours", 9999) < 24)
    metrics["fresh_video_count"] = fresh
    metrics["fresh_video_pct"] = fresh / len(reranked)

    return metrics
```

### Online A/B Test Metrics

| Metric | Direction | Why |
|--------|-----------|-----|
| Total watch time / user / day | Should stay flat or increase | Core engagement |
| Unique creators consumed / user / week | Should increase | Diversity working |
| "Not interested" rate | Should decrease | Better filtering |
| New video impression rate | Should increase | Freshness working |
| Creator upload frequency | Should stay flat or increase | Creator ecosystem health |
| Session count / user / week | Should increase | Exploration driving return visits |

---

## 7. Advanced Topics

### Position Bias in Re-Ranking

Items placed at position 1 naturally get more attention. The re-ranker should account for this:

```python
def position_aware_utility(candidate, position):
    """
    Expected utility = relevance * examination_probability

    Users examine items with decreasing probability down the list.
    """
    examination_prob = 1.0 / np.log2(position + 2)  # Cascade model
    return candidate.ranking_score * examination_prob
```

### Real-Time Context Adaptation

The re-ranker can adapt based on the **current session**:

```python
def session_aware_rerank(candidates, session_history):
    """
    If user has watched 3 gaming videos this session,
    boost other categories to encourage exploration.
    """
    session_categories = Counter(
        v["category"] for v in session_history
    )

    for c in candidates:
        cat = c.video_metadata["category"]
        if session_categories.get(cat, 0) >= 3:
            c.ranking_score *= 0.7  # Reduce score for over-represented category
```

### Fairness Constraints (Creator Side)

```python
def enforce_creator_fairness(candidates, creator_impression_budget):
    """
    Ensure no creator gets more than their fair share of impressions.

    Budget is computed as: total_impressions * creator_content_share * fairness_factor
    """
    for c in candidates:
        creator = c.video_metadata["creator_id"]
        budget = creator_impression_budget.get(creator, float("inf"))
        current = get_creator_impressions_today(creator)

        if current >= budget:
            c.ranking_score *= 0.1  # Heavy penalty, but don't hard-block
```

---

## 8. Interview Talking Points

1. **Why not train an ML model for re-ranking?**
   - Business rules change frequently (new policy = new rule, deployed in hours)
   - ML models need retraining, data collection, A/B testing cycle
   - Hard constraints (safety, legal) must be 100% enforced — ML models can't guarantee this
   - Some teams DO use a lightweight ML re-ranker (e.g., listwise re-ranking with attention), but always with hard filters on top

2. **Diversity vs. relevance trade-off**
   - More diversity = less immediate relevance
   - But long-term: diversity prevents filter bubbles → better retention
   - MMR λ parameter controls this — typically tuned via A/B test
   - YouTube has reported that diversity improvements increased long-term engagement

3. **Exploration: why not just trust the ranking model?**
   - Ranking model is trained on historical data → self-reinforcing loop
   - Videos that never get shown never get training signal → permanently buried
   - 5-10% exploration slots break this feedback loop
   - Exploration also discovers new user interests → better long-term engagement

4. **Latency budget**
   - Re-ranking operates on only ~100 candidates → fast
   - MMR is O(N² × D) but N=100, D=128 → milliseconds
   - Hard filters are just hash lookups → microseconds
   - Total re-ranking budget: < 10ms

5. **How to decide the weights/thresholds?**
   - Start with reasonable defaults from domain knowledge
   - Run A/B tests varying each parameter
   - Monitor both short-term (CTR, watch time) and long-term (DAU retention, creator health) metrics
   - Some companies use multi-armed bandits to auto-tune parameters

6. **Monitoring and rollback**
   - Track all guardrail metrics in dashboards
   - Auto-rollback if: DAU drops > 0.5%, dislike rate increases > 5%, report rate increases > 10%
   - Log every filtering/adjustment decision for debugging ("why was this video not shown?")
