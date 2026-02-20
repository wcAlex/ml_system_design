# Recommendation Systems: A Complete Landscape

## Overview

```
                        Recommendation Systems
                                |
                +---------------+----------------+
                |                                |
         Non-Personalized                  Personalized
                |                                |
        +-------+-------+          +------------+------------+
        |               |          |            |            |
    Popularity      Rule-Based   Content    Collaborative   Hybrid
     -Based                      -Based      Filtering
                                    |            |
                                    |      +-----+------+
                                    |      |            |
                                    |   Memory-     Model-
                                    |   Based       Based
                                    |   (KNN)    (MF, DL)
                                    |
                              +-----+------+
                              |            |
                          Item-Profile   Knowledge
                           Matching      -Based
```

---

## 1. Non-Personalized Recommendations

These methods recommend the **same content to every user**. No user history or profile is needed.

### 1A. Popularity-Based

**How it works:** Rank items by global popularity metrics (views, likes, trending velocity).

```
Examples:
- "Top 10 videos this week"
- "Trending now"
- YouTube's Trending tab
- App store "Top Charts"
```

| Aspect | Detail |
|---|---|
| Input | Aggregate engagement data across all users |
| Output | Same ranked list for everyone |
| Cold-start | Solves user cold-start completely (no history needed) |
| Diversity | Poor -- creates winner-take-all dynamics |
| When to use | New users, fallback when personalization fails, trending sections |

**Variants:**
- **Global popularity**: All-time most viewed
- **Recency-weighted**: Trending in last 24h (decay function on engagement counts)
- **Segment popularity**: Most popular within a region, age group, or device type (slightly personalized)

---

### 1B. Rule-Based / Editorial

**How it works:** Human-defined rules or curated lists determine what to show.

```
Examples:
- "If user is in US, show Super Bowl content during February"
- "If new movie released, pin trailer to top"
- "Show holiday-themed content during December"
- Netflix's curated "collections"
```

| Aspect | Detail |
|---|---|
| Input | Business rules, editorial judgment, time/context signals |
| Output | Deterministic, predictable results |
| Precision | Can be very high for known events/contexts |
| Scalability | Does NOT scale -- requires human effort per rule |
| When to use | Promotions, events, content policy enforcement, bootstrapping |

**Trade-off:** High control, low scalability. Good as a complement, bad as a primary system.

---

## 2. Personalized Recommendations

These methods tailor content to **each individual user** based on their history, profile, or behavior.

---

### 2A. Content-Based Filtering

**Core idea:** Recommend items SIMILAR to what the user has liked before, based on item attributes.

```
User liked: [Action movie with Tom Cruise] [Action movie with Keanu Reeves]
System learns: User likes → action genre, male lead, high-budget
Recommends: Other action movies matching these attributes
```

**How it works:**
1. Build an **item profile** (feature vector) from metadata: genre, tags, description, duration, creator, visual/audio embeddings
2. Build a **user profile** by aggregating item profiles of videos the user engaged with
3. Recommend items whose profiles are most similar to the user profile (cosine similarity, etc.)

```
Item Vector:  [genre=action, duration=long, language=EN, topic=sci-fi, ...]
User Vector:  weighted average of item vectors from watch history
Score:        cosine_similarity(user_vector, candidate_item_vector)
```

| Pros | Cons |
|---|---|
| No need for other users' data (works in isolation) | Limited to user's existing interests (filter bubble) |
| Handles item cold-start well (just need metadata) | Cannot discover cross-genre surprises |
| Transparent / explainable ("because you watched X") | Requires good item features / metadata |
| No popularity bias | Struggles with user cold-start (no history yet) |
| Works even with a single user | Quality depends heavily on feature engineering |

**Best for:** Domains with rich item metadata (news articles, research papers, job postings). Also used as a component in hybrid systems.

---

### 2B. Collaborative Filtering (CF)

**Core idea:** Recommend items that SIMILAR USERS liked. No item features needed -- purely based on user-item interaction patterns.

```
User A watched: [Video 1, Video 2, Video 3]
User B watched: [Video 1, Video 2, Video 4]
Users A and B are similar → Recommend Video 4 to User A, Video 3 to User B
```

#### 2B-i. Memory-Based CF (Neighborhood Methods / KNN)

Directly uses the user-item interaction matrix. No model training.

**User-Based CF:**
1. Find K users most similar to the target user (by interaction overlap)
2. Recommend items those similar users liked but the target user hasn't seen

**Item-Based CF:**
1. Find items most similar to items the target user liked (by co-occurrence in user histories)
2. Recommend the most similar items

```
User-Item Matrix:
              Video1  Video2  Video3  Video4  Video5
    UserA       1       1       1       0       ?
    UserB       1       1       0       1       1
    UserC       0       1       1       0       1

Item-based: Video5 co-occurs with Video2 (liked by A) → recommend Video5 to A
User-based: UserB is similar to UserA → recommend Video4 to A
```

| Pros | Cons |
|---|---|
| Simple, intuitive, explainable | Does NOT scale (O(n^2) similarity computation) |
| No training phase needed | Sparse matrix problem (most users see few items) |
| Item-based CF is quite stable | Cannot handle new users or new items (cold-start) |

**Similarity metrics:** Cosine similarity, Pearson correlation, Jaccard index

---

#### 2B-ii. Model-Based CF

Learn a compact representation (model) from the interaction matrix rather than storing the full matrix.

**Matrix Factorization (MF) -- the classic approach:**

Decompose the user-item interaction matrix into two low-rank matrices:

```
R ≈ U × V^T

R: user-item matrix (m users × n items)
U: user embedding matrix (m × k)     -- k latent factors per user
V: item embedding matrix (n × k)     -- k latent factors per item

Predicted score: r(user, item) = dot_product(U[user], V[item])
```

| Variant | Description |
|---|---|
| SVD / SVD++ | Classic matrix factorization, used in Netflix Prize |
| ALS (Alternating Least Squares) | Good for implicit feedback (clicks, views) |
| BPR (Bayesian Personalized Ranking) | Optimizes pairwise ranking instead of pointwise |
| NMF (Non-negative MF) | Constrains factors to be non-negative for interpretability |

**Deep Learning CF Models:**

| Model | Key Idea |
|---|---|
| Neural Collaborative Filtering (NCF) | Replace dot product with a neural network |
| Autoencoders (e.g., CDAE, MultVAE) | Encode user interaction history, decode to predict missing interactions |
| Two-Tower Model | Separate user encoder and item encoder, dot product for scoring. **This is the industry standard for retrieval** |
| Graph Neural Networks (GNN) | Model user-item interactions as a bipartite graph (PinSage, LightGCN) |

| Pros | Cons |
|---|---|
| Scales well (embeddings are compact) | Cannot use item/user features directly (pure CF) |
| Captures latent patterns humans can't articulate | Cold-start problem remains |
| Proven at scale (Netflix, Spotify, YouTube) | Training can be expensive |
| Handles implicit feedback naturally | Less interpretable than memory-based |

---

### 2C. Knowledge-Based Filtering

**Core idea:** Recommend items based on explicit user requirements or domain knowledge, not interaction history.

```
Examples:
- "I want a video under 5 minutes about Python decorators"
- Real estate: "3 bedrooms, under $500k, near good schools"
- Travel: "Beach destination, budget < $2000, family-friendly"
```

| Pros | Cons |
|---|---|
| No cold-start (works from first interaction) | Requires explicit user input |
| No popularity bias | Doesn't learn preferences over time |
| Precise for constrained domains | Doesn't scale to open-ended browsing |

**When to use:** High-stakes, infrequent decisions (cars, houses, enterprise software). Rarely the primary method for video recommendations but useful for search/filter features.

---

### 2D. Hybrid Approaches

**Core idea:** Combine multiple methods to offset each other's weaknesses. **This is what every production system uses.**

#### Hybrid Strategies

| Strategy | How it works | Example |
|---|---|---|
| **Weighted** | Combine scores from multiple models | `score = 0.6*CF_score + 0.4*content_score` |
| **Switching** | Use different methods depending on context | Use content-based for new users, CF for established users |
| **Cascade** | One method generates candidates, another re-ranks | CF retrieves 1000 candidates → content-based model re-ranks top 50 |
| **Feature Augmentation** | Output of one model becomes input feature for another | CF embeddings fed as features into a content-based neural ranker |
| **Meta-Level** | One model's learned representation is input to another | Use CF to learn user embeddings, then use those in a content-aware model |

#### The Modern Production Pipeline (YouTube, TikTok, Instagram)

This is the hybrid approach used at scale. It's a **cascade / multi-stage pipeline**:

```
Stage 1: CANDIDATE GENERATION (Retrieval)
├── Source A: Collaborative Filtering (two-tower model, ~100 candidates)
├── Source B: Content-Based (similar to recently watched, ~100 candidates)
├── Source C: Popularity (trending videos, ~50 candidates)
├── Source D: Social (what friends watched, ~50 candidates)
├── Source E: Subscriptions (new uploads from followed creators, ~50 candidates)
│
├── Merge & Deduplicate → ~300 unique candidates
│
Stage 2: RANKING
├── A single deep ranking model scores all candidates
├── Uses BOTH CF features (user/item embeddings) AND content features
├── Multi-task: predicts P(click), E[watch_time], P(like), P(finish)
├── Combines into satisfaction score
│
Stage 3: RE-RANKING (Post-Processing)
├── Apply guardrails (diversity, freshness, safety)
├── Final list of ~20-50 videos shown to user
```

**This is the answer interviewers want.** It's hybrid at every stage:
- Retrieval mixes CF + content-based + popularity
- Ranking uses both CF embeddings and content features in one model
- Re-ranking applies rule-based guardrails

---

## Comparison Matrix

| Method | Personalized | Cold-Start (User) | Cold-Start (Item) | Scalability | Serendipity | Interpretability |
|---|---|---|---|---|---|---|
| Popularity | No | Handles well | Handles poorly | Excellent | Low | High |
| Rule-Based | No | Handles well | Handles well | Poor (manual) | Low | High |
| Content-Based | Yes | Poor | Good | Good | Low | High |
| CF (Memory) | Yes | Poor | Poor | Poor | Medium | Medium |
| CF (Model/MF) | Yes | Poor | Poor | Good | Medium | Low |
| CF (Deep Learning) | Yes | Medium* | Medium* | Good | Medium | Low |
| Knowledge-Based | Yes | Good | Good | Medium | Low | High |
| Hybrid Pipeline | Yes | Good | Good | Excellent | High | Medium |

*Deep learning CF can incorporate side features (user demographics, item metadata) to partially address cold-start.

---

## Trade-Off Summary: The Key Tensions

### 1. Exploitation vs. Exploration
- **Exploitation**: Recommend what the model is confident the user will like (safe, high engagement)
- **Exploration**: Recommend uncertain items to discover new interests (risky, data collection)
- **Resolution**: Epsilon-greedy, Thompson sampling, or dedicated explore slots

### 2. Accuracy vs. Diversity
- **Accuracy**: Show the most relevant items (but they'll all look the same)
- **Diversity**: Show varied items (but some will be less relevant)
- **Resolution**: MMR re-ranking, category caps, diversity-aware loss functions

### 3. Personalization vs. Freshness
- **Personalization**: Favor items with lots of engagement data (older, proven items)
- **Freshness**: Favor new items (less data, higher risk)
- **Resolution**: Time-decay features, freshness boost slots, explore-exploit for new items

### 4. Short-Term vs. Long-Term Engagement
- **Short-term**: Optimize for this session (clickbait can win)
- **Long-term**: Optimize for user returning next week (requires satisfaction, not just clicks)
- **Resolution**: Use multi-objective scoring (click + watch time + likes - dislikes), monitor retention as guardrail

### 5. Simplicity vs. Performance
- **Simple models**: Easy to debug, ship, and maintain. MF can be surprisingly strong.
- **Complex models**: Better offline metrics, but harder to iterate, more infra needed.
- **Resolution**: Start simple (MF / two-tower), add complexity only when you can measure improvement via A/B tests

### 6. Privacy vs. Personalization
- **More data**: Better recommendations
- **Privacy**: Users may not want their full history used, regulations (GDPR) limit data retention
- **Resolution**: On-device personalization, differential privacy, federated learning, data retention policies

---

## Interview Cheat Sheet

When asked "How would you build a recommendation system?", structure your answer as:

1. **Start with the pipeline**: "I'd use a multi-stage pipeline: retrieval → ranking → re-ranking"
2. **Retrieval**: "Multiple candidate sources -- CF two-tower model for personalized recall, content-based for item cold-start, popularity for user cold-start"
3. **Ranking**: "A deep ranking model that combines CF embeddings with content/context features, multi-task trained on click, watch time, and like objectives"
4. **Re-ranking**: "Post-processing for diversity, freshness, and safety guardrails"
5. **Mention the trade-offs**: Show you understand that each choice has a cost

This structure covers all paradigms (CF, content-based, popularity, rules) in a unified, practical architecture.
