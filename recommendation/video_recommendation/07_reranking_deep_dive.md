# Re-Ranking: From Basics to Production

A ground-up explanation of why ranking isn't enough, and how re-ranking transforms a ranked list into the final recommendations users actually see — using YouTube and Netflix as real-world examples.

---

## 1. What Is Re-Ranking and Why Do We Need It?

The ranking model outputs a list sorted by predicted engagement score. Sounds like we're done? Not even close.

```
Ranking model output (sorted by score):
  #1  Gaming video from Creator X     (score: 9.2)
  #2  Gaming video from Creator X     (score: 8.8)
  #3  Gaming video from Creator X     (score: 8.5)
  #4  Gaming video from Creator Y     (score: 8.3)
  #5  Gaming video from Creator X     (score: 8.1)
  #6  Gaming video from Creator Z     (score: 7.9)
  #7  Gaming video from Creator X     (score: 7.7)
  #8  Cooking video from Creator W    (score: 7.5)
  ...

Problems:
  ✗ 5 of top 7 are from the same creator (Creator X dominance)
  ✗ All top videos are gaming (no diversity)
  ✗ No fresh/new content (all established videos with historical data)
  ✗ Possibly includes a video the user already watched yesterday
  ✗ Possibly includes a video that violates content policy
```

The ranking model optimizes for **individual item relevance**. But a good recommendation page optimizes for the **quality of the entire list**. These are fundamentally different objectives.

**Real-world analogy**: A restaurant menu ranked purely by "most popular dish" would put burgers as items #1 through #8. A good menu design shows variety — an appetizer, different proteins, vegetarian options — even if burgers are individually the most popular.

---

## 2. Building Up from Simple to Complex

### Level 1: Hard Filters (Bare Minimum)

The simplest re-ranking: remove things that should never be shown.

```
Before filters (from ranking model):
  #1  Video A (score: 9.2)
  #2  Video B (score: 8.8)  ← user watched this yesterday
  #3  Video C (score: 8.5)  ← removed for policy violation
  #4  Video D (score: 8.3)
  #5  Video E (score: 8.1)  ← age-restricted, user is 15

After filters:
  #1  Video A (score: 9.2)
  #2  Video D (score: 8.3)
  #3  (need to fill gaps with lower-ranked items)
```

**Every production system has this layer. Non-negotiable.**

Filters include:
- Already watched by this user
- Policy violations (nudity, violence, misinformation)
- Age-restricted content for underage users
- Region-restricted content
- Blocked creators (user's block list)
- Copyright-claimed content

### Level 2: Simple Diversity Rules (Heuristic)

Add basic constraints to prevent the ranking model from being too monotonous.

```
Rules:
  - Max 2 videos from the same creator in any 10-video window
  - Max 3 videos from the same category in any 10-video window
  - At least 1 video uploaded in last 24 hours in every 5-video window

Before rules:
  #1  Gaming - Creator X
  #2  Gaming - Creator X     ← violates creator cap
  #3  Gaming - Creator Y
  #4  Gaming - Creator Z     ← violates category cap (4 gaming in a row)
  #5  Gaming - Creator X     ← violates creator cap

After rules (push violations down, pull others up):
  #1  Gaming  - Creator X
  #2  Gaming  - Creator Y
  #3  Cooking - Creator W    ← pulled up for category diversity
  #4  Gaming  - Creator Z
  #5  Music   - Creator V    ← pulled up for diversity
```

**This is what most systems start with. Simple, debuggable, effective.**

### Level 3: Score Adjustments (Soft Penalties & Boosts)

Instead of hard rules, modify the ranking score to influence ordering.

```
Original score: 8.5

Adjustments applied:
  × 1.20 freshness boost (uploaded 2 hours ago)
  × 0.85 clickbait penalty (high P(click) but low P(finish))
  × 1.10 quality boost (95% like ratio)
  × 1.00 no creator penalty (first video from this creator)

Adjusted score: 8.5 × 1.20 × 0.85 × 1.10 × 1.00 = 9.52

The adjusted score determines the new position in the list.
```

This is more nuanced than hard rules: instead of "block the 3rd video from Creator X," you say "reduce the score by 15% for each additional video from the same creator." The video can still appear if its relevance is high enough.

### Level 4: Diversity Optimization (MMR)

**Maximal Marginal Relevance** — a principled algorithm that balances relevance and diversity when building a list.

```
The core idea:

Greedy selection:
  1. Pick the highest-scoring item
  2. For each remaining item, compute:
     MMR(item) = λ × relevance(item) - (1-λ) × max_similarity(item, already_selected)
  3. Pick the item with highest MMR score
  4. Repeat until list is full

λ controls the trade-off:
  λ = 1.0 → pure relevance (same as ranking model output)
  λ = 0.5 → balanced relevance and diversity
  λ = 0.0 → pure diversity (maximally spread out items)
```

**Worked example**:

```
Already selected: ["PyTorch Tutorial"]
Candidates remaining:

  "TensorFlow Tutorial"
    relevance = 8.5
    similarity to selected = 0.92 (very similar — both ML tutorials)
    MMR = 0.7 × 8.5 - 0.3 × 0.92 = 5.95 - 0.276 = 5.67

  "Minecraft Stream"
    relevance = 7.0
    similarity to selected = 0.15 (very different — gaming vs ML)
    MMR = 0.7 × 7.0 - 0.3 × 0.15 = 4.90 - 0.045 = 4.86

  "Neural Network Math"
    relevance = 8.0
    similarity to selected = 0.85 (similar — both ML)
    MMR = 0.7 × 8.0 - 0.3 × 0.85 = 5.60 - 0.255 = 5.35

Winner: "TensorFlow Tutorial" (highest MMR: 5.67)
  But notice: "Minecraft Stream" moved from rank 3 to rank 2 candidate
  because it provides diversity even though its relevance is lower.
```

### Level 5: Exploration Slots (Breaking the Feedback Loop)

Reserve a fraction of slots for **exploration** — candidates the model is uncertain about.

```
Final recommendation list (30 slots):
  Slots 1-26:  Ranked by adjusted score + diversity (exploitation)
  Slots 27-29: Exploration candidates (randomly sampled from lower ranks)
  Slot 30:     Fresh content slot (video uploaded < 6 hours ago)

Why exploration matters:
  - The ranking model is trained on PAST data
  - It can only recommend what it's seen work before
  - Without exploration, new videos and new interests never get discovered
  - Users get stuck in a "filter bubble" of their existing preferences
```

---

## 3. YouTube's Re-Ranking in Practice

### What YouTube Optimizes For (Beyond Relevance)

YouTube has published about their "responsible recommendations" work. Their re-ranking considers:

```
1. Satisfaction signals (from ranking model)
   - Predicted watch time, likes, shares

2. Content quality (re-ranking adjustments)
   - Authoritative sources for news/health topics
   - Down-rank borderline content (not policy-violating but low quality)
   - Creator authority score

3. Diversity (re-ranking rules)
   - Topic diversity: mix of categories
   - Creator diversity: cap per creator
   - Format diversity: mix of shorts, long-form, live

4. Freshness (slot reservation)
   - Ensure recent uploads get exposure
   - Critical for the creator ecosystem

5. User controls (hard filters)
   - "Not interested" feedback → remove similar content
   - "Don't recommend channel" → block creator
   - Watch history on/off toggle
```

### YouTube's "Breaking the Filter Bubble" Approach

```
YouTube explicitly reserves recommendation slots for:

1. "Explore" candidates
   - From topics the user hasn't watched but might enjoy
   - Based on interests of similar users

2. "Authoritative" candidates
   - For news/health queries, boost authoritative sources
   - Even if they have lower engagement metrics

3. "New creator" candidates
   - From creators with < 1000 subscribers
   - Ensures new creators can get discovered

4. "Fresh" candidates
   - Videos uploaded in last 24 hours
   - Ensures the platform feels current
```

---

## 4. Netflix's Re-Ranking in Practice

Netflix's re-ranking is uniquely interesting because they optimize for **rows and pages**, not just a single feed.

### Netflix's Row-Based Layout

```
Netflix Home Page:

  Row 1: "Continue Watching"      [show_A] [show_B] [show_C]
  Row 2: "Trending Now"           [show_D] [show_E] [show_F]
  Row 3: "Because You Watched X"  [show_G] [show_H] [show_I]
  Row 4: "New Releases"           [show_J] [show_K] [show_L]
  Row 5: "Top 10 in Your Country" [show_M] [show_N] [show_O]

Each row has its own ranking model.
But the PAGE LAYOUT itself is also optimized:
  - Which rows to show
  - In what order
  - How many items per row
```

### Netflix's Re-Ranking Considerations

```
1. Cross-row deduplication
   A title should not appear in multiple rows on the same page.
   If "Stranger Things" qualifies for "Trending" AND "Because You Watched",
   show it in whichever row the user is more likely to notice.

2. Row diversity
   Don't show 5 comedy rows in a row.
   Alternate genres: action, comedy, drama, documentary, ...

3. "Evidence" diversity
   Mix the reasoning behind recommendations:
   - Social proof: "Trending in your country"
   - Personalization: "Because you watched X"
   - Novelty: "New releases"
   - Nostalgia: "Watch again"
   Users trust recommendations more when the REASONS are diverse.

4. Page-level freshness
   Ensure the page looks different each time the user visits.
   If the same titles are always on top, it feels stale.
   Netflix shuffles within confidence intervals to provide variety.
```

### Netflix's "Explore-Exploit" Balance

Netflix uses a **multi-armed bandit** approach to exploration:

```
Traditional: Show the highest-scoring item always (exploitation)
  → Optimal in the short term but creates filter bubbles

Netflix: Thompson Sampling
  1. For each item, maintain a distribution over its true quality
     (not just a point estimate)
  2. Sample from each distribution
  3. Rank by sampled values

  Item A: estimated P(play) = 0.30, uncertainty = ±0.02 (well-known)
  Item B: estimated P(play) = 0.25, uncertainty = ±0.15 (uncertain)

  Sample from A: 0.31
  Sample from B: 0.38  ← uncertainty caused a high sample!

  → Item B gets shown, user either plays or doesn't
  → Update the distribution for B → uncertainty shrinks
  → Next time, B's estimate is more accurate

  This naturally balances:
  - Showing known-good items (exploitation)
  - Testing uncertain items (exploration)
  - Exploration decreases over time as uncertainty shrinks
```

---

## 5. The Full Re-Ranking Pipeline (Production)

Here's how everything fits together in a complete system:

```
Input: ~100 scored candidates from the ranking model
                │
                ▼
┌───────────────────────────────────┐
│  LAYER 1: HARD FILTERS            │
│                                    │
│  Remove:                           │
│  ✗ Already watched                 │
│  ✗ Policy violations               │
│  ✗ Age-restricted (if underage)    │
│  ✗ Blocked creators                │
│  ✗ Region-restricted               │
│                                    │
│  ~100 → ~85 candidates             │
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  LAYER 2: SCORE ADJUSTMENTS       │
│                                    │
│  For each candidate:               │
│  × Clickbait penalty (-30%)        │
│  × Quality multiplier (+15%)       │
│  × Freshness boost (+20%)          │
│  × Creator fatigue (-10% per       │
│    additional video from same      │
│    creator already in list)        │
│                                    │
│  Re-sort by adjusted scores        │
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  LAYER 3: DIVERSITY (MMR)          │
│                                    │
│  Greedy selection with MMR:        │
│  λ = 0.7 (mostly relevance,       │
│           some diversity)          │
│                                    │
│  Select top 35 items               │
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  LAYER 4: SLOT CONSTRAINTS         │
│                                    │
│  Hard caps:                        │
│  • Max 2 per creator               │
│  • Max 4 per category              │
│  • At least 1 fresh video          │
│                                    │
│  Swap out violations, pull in      │
│  next-best candidates              │
│                                    │
│  35 → 32 after swaps               │
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  LAYER 5: EXPLORATION              │
│                                    │
│  Reserve 3 slots for exploration:  │
│  • 1 slot: under-exposed video     │
│  • 1 slot: new creator             │
│  • 1 slot: different topic area    │
│                                    │
│  Insert at random positions in     │
│  the bottom 2/3 of the list        │
│                                    │
│  32 + 3 = 35 → trim to 30 final   │
└───────────────┬───────────────────┘
                │
                ▼
        Final 30 recommendations
        (shown to the user)
```

---

## 6. Measuring Re-Ranking Quality

Re-ranking is harder to evaluate than ranking because it optimizes for **list-level properties**, not individual item accuracy.

### Offline Metrics

| Metric | What It Measures | How to Compute |
|--------|-----------------|----------------|
| **NDCG degradation** | How much relevance was sacrificed for diversity | NDCG(before re-ranking) - NDCG(after re-ranking). Should be < 5% |
| **Intra-List Diversity (ILD)** | How different items are from each other | Average pairwise cosine distance between all recommended items |
| **Category entropy** | How spread across categories the list is | Shannon entropy of category distribution. Higher = more diverse |
| **Coverage** | What % of the catalog gets recommended | Count unique items across all users / total catalog. Higher = less filter bubble |
| **Creator Gini coefficient** | How evenly impressions are distributed across creators | 0 = perfectly even, 1 = all impressions to one creator |
| **Freshness ratio** | % of recommendations that are recent | Count items < 24h old / total. Target: 5-15% |

### Online Metrics (A/B Test)

| Metric | Expected Direction | Why |
|--------|-------------------|-----|
| Total watch time / user / day | Flat or slight increase | Diversity can increase session variety → longer sessions |
| Unique videos watched / user / week | Increase | Diversity exposes users to more content |
| Unique creators consumed / user / week | Increase | Creator cap ensures exposure to more creators |
| "Not interested" rate | Decrease | Better filtering reduces irrelevant content |
| DAU retention (7-day) | Flat or increase | Long-term health metric |
| New creator impression share | Increase | Exploration slots working |

### The Key Trade-Off to Monitor

```
                Relevance
                    ▲
                    │
              100%  │ ● No re-ranking
                    │    (pure ranking model output)
                    │
               97%  │         ● Light re-ranking
                    │           (filters + basic diversity)
               95%  │                ● Moderate re-ranking
                    │                   (MMR + slot constraints)
               90%  │                        ● Heavy re-ranking
                    │                           (aggressive diversity)
                    │
                    └─────────────────────────────────▶ Diversity
                      Low                           High

"Sweet spot" is typically 3-5% relevance sacrifice for
meaningful diversity improvement (verified via A/B test).
```

---

## 7. Why Not Train the Re-Ranking Rules into the Ranking Model?

This is a common interview question. The answer has several parts:

### Reason 1: Non-Differentiable Constraints

```
Rule: "Max 2 videos per creator"

This is a HARD constraint — binary, all-or-nothing.
You can't express "max 2 per creator" as a differentiable loss function.
Neural networks optimize continuous losses with gradient descent.
Hard constraints don't have gradients.
```

### Reason 2: Iteration Speed

```
Re-ranking rule change:
  Day 1: Product team says "cap creator to 3 videos"
  Day 1: Engineer changes config: max_per_creator = 3
  Day 1: Deployed to production
  → Live in hours

Ranking model change:
  Day 1: Data scientist modifies loss function
  Day 2-3: Retrain model (hours to days)
  Day 3-5: Offline evaluation
  Day 5-7: Canary deployment
  Day 7-14: A/B test
  Day 14: Full rollout
  → Live in weeks
```

### Reason 3: Organizational Separation

```
Ranking model: Owned by ML team
  → Changes require careful evaluation, can break engagement

Re-ranking rules: Owned by product/policy team
  → Changes driven by business strategy, legal, PR
  → Need to iterate independently of ML training cycles

Example: New content policy (e.g., COVID misinformation)
  → Must be enforced TODAY, not after 2 weeks of model retraining
  → Re-ranking filter: deploy in hours
```

### Reason 4: Explainability

```
When user complains: "Why did you show me this?"

Re-ranking rules: "Because it was in your top 30 after
  filtering already-watched and ensuring diversity."
  → Easy to trace, debug, explain

Ranking model: "Because the neural network predicted
  a combined score of 7.32 based on 700 features."
  → Much harder to explain

When regulators ask: "How do you ensure fairness?"

Re-ranking rules: "We cap each creator at 2 videos per
  recommendation page and reserve slots for new creators."
  → Clear, auditable, documented

Ranking model: "We hope the model learned fairness from the data."
  → Not auditable, not explainable
```

---

## 8. Advanced Re-Ranking Techniques

### Technique 1: Listwise Re-Ranking with Attention

Instead of greedy MMR, use a **neural network** that takes the entire ranked list and re-orders it.

```
Input: ranked list of N items (each with features + ranking score)

  [item_1 features] [item_2 features] ... [item_N features]
           │                │                      │
           └────────────────┼──────────────────────┘
                            │
                   Self-Attention Layers
                   (items attend to each other)
                            │
           ┌────────────────┼──────────────────────┐
           │                │                      │
      new_score_1      new_score_2          new_score_N

The attention mechanism can learn:
  - "If item 3 is very similar to item 1, reduce item 3's score"
  - "If no cooking videos in top 5, boost the first cooking video"
  - Implicit diversity without explicit rules
```

**Trade-off**: Better quality than heuristic rules, but harder to debug and explain. Used at some large companies alongside (not replacing) rule-based re-ranking.

### Technique 2: Contextual Bandits for Exploration

More sophisticated than random exploration: use a **contextual bandit** to decide when and what to explore.

```
Traditional exploration: randomly insert 3 items into the list

Contextual bandit:
  Input: (user features, session context, current list composition)
  Output: which item to explore and at which position

  The bandit learns:
  - "New users benefit from more exploration (wider net)"
  - "Users in long sessions are open to more variety"
  - "Don't explore on the very first position (too risky)"
```

### Technique 3: Real-Time Session Adaptation

Adapt the re-ranking based on what happened earlier in the same session.

```
Session starts:
  User watches 3 gaming videos in a row
  Re-ranker detects: "session is gaming-heavy"
  Action: boost non-gaming categories by 20%

  User then watches a cooking video
  Re-ranker: "user is now open to other topics"
  Action: increase diversity λ in MMR

This is more responsive than the ranking model, which uses
features from BEFORE the session started.
```

---

## 9. Common Pitfalls

### Pitfall 1: Over-Diversifying

```
Problem: Making the list SO diverse that nothing feels relevant.

  User loves gaming. Recommendation page:
    #1 Gaming video
    #2 Cooking video (for diversity)
    #3 News video (for diversity)
    #4 Music video (for diversity)
    #5 Fitness video (for diversity)

  User: "This doesn't feel personalized at all."

Fix: Set diversity constraints to a moderate level.
  Typically cap at 3-5% NDCG degradation vs pure ranking.
  A/B test to find the sweet spot.
```

### Pitfall 2: Conflicting Rules

```
Problem: Multiple rules that contradict each other.

  Rule 1: "At least 2 fresh videos in top 10"
  Rule 2: "Max 3 from same category in top 10"
  Rule 3: "At least 1 from each of user's top 3 categories in top 10"

  What if the only fresh videos are all in the same category?
  Rule 1 and Rule 2 conflict.

Fix: Establish rule priority ordering:
  Priority 1: Safety filters (absolute)
  Priority 2: Freshness slots (business critical)
  Priority 3: Creator caps (fairness)
  Priority 4: Category diversity (nice to have)
  Lower-priority rules yield to higher-priority ones.
```

### Pitfall 3: Not Measuring the Cost

```
Problem: Adding re-ranking rules without measuring relevance impact.

  Team adds 10 rules over 6 months.
  Each rule costs ~0.5% NDCG.
  Combined: 5% NDCG degradation — significant engagement drop.
  Nobody noticed because rules were added one at a time.

Fix: Dashboard showing NDCG before vs after re-ranking.
  Alert if cumulative degradation exceeds threshold (e.g., 5%).
  Periodically audit: is each rule still needed?
```

---

## 10. Interview Cheat Sheet: Re-Ranking

### "Why do you need re-ranking if you already have a ranking model?"

> The ranking model scores each item independently — it doesn't consider what else is in the list. But users see a **list**, not individual items. Re-ranking optimizes for list-level properties that the ranking model can't: diversity across creators and topics, freshness, safety filtering, creator fairness, and exploration for new content. These constraints are also non-differentiable (hard caps, binary filters), so they can't be expressed as training losses.

### "Walk me through your re-ranking approach"

> Five layers, each with a clear purpose:
> 1. **Hard filters**: Remove already-watched, policy violations, age/region restrictions. Non-negotiable.
> 2. **Score adjustments**: Soft penalties for clickbait and quality; soft boosts for freshness. Modifies ranking score by multipliers.
> 3. **Diversity via MMR**: Greedy selection balancing relevance (λ=0.7) and diversity. The candidate's score is penalized by its similarity to already-selected items.
> 4. **Slot constraints**: Hard caps on per-creator and per-category counts. Swap violations with next-best alternatives.
> 5. **Exploration**: Reserve 5-10% of slots for under-exposed content, new creators, and different topics. Inserted in the bottom 2/3 of the list.

### Common follow-up questions

| Question | Key Point |
|----------|-----------|
| "How do you measure if re-ranking helps?" | A/B test: watch time (should be flat), diversity metrics (should increase), "not interested" rate (should decrease) |
| "How much relevance do you sacrifice?" | Target < 5% NDCG degradation. Monitor continuously |
| "Why not train diversity into the model?" | Non-differentiable constraints, need fast iteration, organizational separation, explainability |
| "How do you handle conflicting rules?" | Priority ordering: safety > freshness > fairness > diversity. Lower-priority rules yield |
| "What about using a learned re-ranker?" | Listwise attention models exist but are used alongside (not replacing) rules. Rules handle hard constraints; learned models handle soft quality optimization |
| "How does Netflix do it differently?" | Netflix optimizes rows + pages (not just a feed). Cross-row dedup, row ordering, evidence diversity, Thompson Sampling for exploration |

---

## 11. Relationship to Other Components

```
For implementation details (code, data structures):
  → models/04_reranking/README.md

For the serving infrastructure:
  → 05_serving_system.md (re-ranking runs in-process in the orchestrator, <15ms)

For what feeds INTO re-ranking:
  → 06_ranking_deep_dive.md
```
