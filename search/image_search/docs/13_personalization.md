# 13 — Personalization

## Why Personalization Matters in Visual Search

Two users upload the exact same photo of a shoe. One is a 22-year-old who buys streetwear; the other is a 45-year-old professional who buys formal wear. Their ideal results are different.

**Visual similarity is necessary but not sufficient.** Personalization bridges the gap between "visually relevant" and "personally relevant."

---

## What Personalization Signals Tell Us

| Signal | Type | Recency | Strength |
|---|---|---|---|
| Purchase history | Long-term preference | Historical | Very strong (real intent) |
| Search + click history | Long-term preference | Historical | Strong (expressed interest) |
| Wishlist / saved items | Long-term preference | Historical | Strong |
| Recently viewed items | Short-term (session) | Current session | Medium (recency bias) |
| Cart items | Short-term (session) | Current session | Strong (purchase intent) |
| Category browsing | Long-term preference | Last 30 days | Medium |
| Price tier of past purchases | Long-term preference | Historical | Medium |
| Brand affinity | Long-term preference | Historical | Medium |

---

## Personalization Architecture: Three Levels

### Level 1: Rule-Based Personalization (Simplest, Fastest)

Apply simple business rules post-retrieval:

```python
def rule_based_personalization(candidates, user_profile):
    """
    Apply simple multipliers based on user history.
    No ML required. Fast to build, easy to debug.
    """
    for item in candidates:
        score_multiplier = 1.0

        # Boost items from preferred categories
        if item["category"] in user_profile.get("preferred_categories", []):
            score_multiplier *= 1.15

        # Boost items from preferred brands
        if item["brand"] in user_profile.get("preferred_brands", []):
            score_multiplier *= 1.10

        # Penalize items far from typical spend
        typical_spend = user_profile.get("avg_purchase_price", 100)
        price_ratio = item["price"] / (typical_spend + 1e-6)
        if price_ratio > 2.0 or price_ratio < 0.3:
            score_multiplier *= 0.85

        # Suppress recently purchased items (avoid duplicates)
        if item["product_id"] in user_profile.get("recent_purchases", set()):
            score_multiplier *= 0.5  # strong suppression

        item["final_score"] = item["visual_score"] * score_multiplier

    return sorted(candidates, key=lambda x: x["final_score"], reverse=True)
```

**Pros:** Transparent, debuggable, no training required, works immediately
**Cons:** Misses non-linear interactions, doesn't adapt to individual nuances

---

### Level 2: User Embedding Personalization (Two-Tower Model)

Learn a user embedding that captures long-term preferences, combine with query embedding.

#### Architecture

```
Query Side (online)                    Product Side (offline, precomputed)
─────────────────────                  ─────────────────────────────────
                                       Product Image → Image Encoder
User ID                                     │
   │                                        ▼
   ▼                                   512-d product visual embedding
User Tower                                  │
(MLP on user features)                      │
   │                                        │
   ▼                                        │
128-d user embedding                        │
   │                                        │
   │ (concatenate or add)                   │
   │                                        │
Query Image → Image Encoder                 │
   │                                        │
   ▼                                        │
512-d query visual embedding                │
   │                                        │
   └──────────────┬─────────────────────────┘
                  │
             Fusion Layer
          (query + user embed)
                  │
                  ▼
           ANN Search or Re-ranking score
```

#### User Tower Architecture

```python
class UserTower(nn.Module):
    """
    Maps user features to a dense user preference embedding.
    Used to modulate visual search results with personal preferences.
    """

    def __init__(self, num_categories=500, num_brands=10000, embed_dim=128):
        super().__init__()
        # Sparse feature embeddings
        self.category_embed = nn.Embedding(num_categories, 32)
        self.brand_embed = nn.Embedding(num_brands, 32)

        # Dense feature processing
        self.price_proj = nn.Linear(1, 16)  # avg purchase price

        # Combine all user signals
        self.mlp = nn.Sequential(
            nn.Linear(32 + 32 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, category_ids, brand_ids, avg_price):
        """
        category_ids: (B, K) — top-K purchased categories, padded
        brand_ids: (B, K) — top-K purchased brands, padded
        avg_price: (B, 1) — normalized average purchase price
        """
        # Mean-pool multi-hot category/brand embeddings
        cat_embed = self.category_embed(category_ids).mean(dim=1)   # (B, 32)
        brand_embed = self.brand_embed(brand_ids).mean(dim=1)        # (B, 32)
        price_embed = self.price_proj(avg_price)                     # (B, 16)

        user_embed = self.mlp(torch.cat([cat_embed, brand_embed, price_embed], dim=-1))
        return F.normalize(user_embed, dim=-1)


class PersonalizedSearcher(nn.Module):
    """Combines image query embedding with user embedding for personalized search."""

    def __init__(self, image_encoder, user_tower, image_dim=512, user_dim=128):
        super().__init__()
        self.image_encoder = image_encoder
        self.user_tower = user_tower

        # Projection to unified search space
        self.query_proj = nn.Linear(image_dim + user_dim, image_dim)

    def get_search_embedding(self, query_image_tensor, user_features):
        img_embed = F.normalize(self.image_encoder(query_image_tensor), dim=-1)
        user_embed = self.user_tower(**user_features)

        # Fuse image + user context
        fused = torch.cat([img_embed, user_embed], dim=-1)
        search_embed = F.normalize(self.query_proj(fused), dim=-1)
        return search_embed
```

#### Training the User Tower

Train jointly with the image encoder using:
- **Positive pairs:** (query image, product that user actually purchased)
- **Negative pairs:** (query image, products shown but not clicked)

This teaches the user tower to shift the embedding toward the user's preferences.

**Cold-start problem:** New users have no history → fall back to Level 1 (rule-based) or use demographic signals.

---

### Level 3: Session-Aware Re-ranking (Short-Term Context)

Beyond long-term user profiles, incorporate what the user is doing *right now* in this session.

```python
class SessionReRanker:
    """
    Re-ranks results based on the user's actions in the current session.
    Uses recency-weighted session history to capture short-term intent shifts.
    """

    def __init__(self, decay_factor: float = 0.8):
        self.decay_factor = decay_factor

    def compute_session_embedding(
        self,
        session_events: list,  # list of {embedding, action_type, timestamp}
        item_embeddings: dict  # product_id → embedding
    ) -> np.ndarray:
        """
        Compute a session-level context vector by aggregating recent interactions.
        More recent events get higher weight.
        """
        if not session_events:
            return None

        weights = [self.decay_factor ** i for i in range(len(session_events) - 1, -1, -1)]
        weighted_embeds = []

        for event, w in zip(session_events, weights):
            embed = event.get("embedding")
            if embed is not None:
                action_weight = {
                    "purchase": 2.0,
                    "add_to_cart": 1.5,
                    "click": 1.0,
                    "view": 0.5
                }.get(event["action_type"], 0.5)
                weighted_embeds.append(w * action_weight * embed)

        if not weighted_embeds:
            return None

        session_embed = np.sum(weighted_embeds, axis=0)
        return session_embed / np.linalg.norm(session_embed)

    def rerank_with_session(
        self,
        candidates: list,
        product_embeddings: dict,
        session_embed: np.ndarray,
        session_weight: float = 0.3
    ) -> list:
        """
        Boost items similar to the session context.
        """
        if session_embed is None:
            return candidates

        for candidate in candidates:
            pid = candidate["product_id"]
            prod_embed = product_embeddings.get(pid)
            if prod_embed is not None:
                session_sim = float(np.dot(session_embed, prod_embed))
                candidate["final_score"] = (
                    (1 - session_weight) * candidate.get("visual_score", 0)
                    + session_weight * session_sim
                )

        return sorted(candidates, key=lambda x: x.get("final_score", 0), reverse=True)
```

---

## Advanced: Exploration vs. Exploitation (Contextual Bandits)

Pure exploitation (always serve the user's top preferences) leads to **filter bubbles** and **discovery failure**. Users see only what they've seen before.

### Epsilon-Greedy Exploration

```python
class ExplorationReranker:
    """
    With probability epsilon, insert an exploratory item to help
    the system learn new preferences and expose users to new products.
    """

    def __init__(self, epsilon: float = 0.05):
        self.epsilon = epsilon

    def apply_exploration(self, ranked_results: list, explore_pool: list) -> list:
        import random
        if random.random() < self.epsilon and explore_pool:
            # Replace one result (not position 1) with an exploratory item
            explore_item = random.choice(explore_pool)
            explore_item["is_exploratory"] = True
            insert_pos = random.randint(3, min(10, len(ranked_results)))
            ranked_results.insert(insert_pos, explore_item)
        return ranked_results
```

### Upper Confidence Bound (UCB) for New Products

For new products with no interaction data, UCB provides a principled way to decide how much to explore:

```
UCB_score(item) = expected_relevance(item) + β × sqrt(log(N) / n_shown(item))

where:
  expected_relevance = visual_similarity × user_preference_score
  n_shown(item) = number of times item was shown to users
  N = total impressions
  β = exploration weight (tune via experiment)
```

Items with fewer impressions get a higher exploration bonus → ensures all products eventually get evaluated.

### LinUCB (Contextual Bandit — Industry Approach)

Yahoo and LinkedIn use contextual bandits for personalized ranking. The model learns which features predict click probability for each user-item pair, with uncertainty-based exploration.

```
reward_estimate(user, item) = θᵀ × features(user, item)
uncertainty_bonus = α × sqrt(features(user, item)ᵀ × A_inv × features(user, item))
UCB_score = reward_estimate + uncertainty_bonus
```

**Why this works:** The uncertainty term is large when features haven't been seen much → automatic exploration for novel situations.

---

## Personalization Cold-Start Strategies

| User State | Strategy | Signals Used |
|---|---|---|
| **New user (no history)** | Trending + diverse results | Category popularity, trending items |
| **<10 interactions** | Lightweight collaborative filtering | Similar users' behavior |
| **Active user (>100 interactions)** | Full user tower personalization | User embedding + session context |
| **Lapsed user (>90 days inactive)** | Ignore stale history, treat as semi-new | Recent trend signals + sparse old history |

---

## Measuring Personalization Quality

| Metric | What It Measures |
|---|---|
| **Personalized CTR lift** | CTR improvement vs. non-personalized baseline (A/B test) |
| **Diversity@K** | Average pairwise distance between top-K results (avoid echo chambers) |
| **Serendipity score** | Fraction of highly clicked items that were "surprising" (low prior affinity) |
| **Repeat purchase rate** | Did personalized results lead to repeat engagement? |
| **User satisfaction survey** | "Did you find what you were looking for?" (sampled) |

---

## Industry Examples

| Company | Personalization Approach |
|---|---|
| **Pinterest** | User interest embeddings + session graph; "Board" context |
| **Amazon StyleSnap** | Purchase/wishlist history + outfit completion model |
| **Google Shopping Lens** | Shopping graph: past searches, location, price sensitivity |
| **TikTok Shop** | Reinforcement learning over session: optimize for watch time → purchase |

---

## Personalization Summary: Three Phases

```
Phase 1 (Launch):    Rule-based personalization — category/brand boost, price filtering
Phase 2 (Growth):    User tower in two-tower model — trained on click/purchase data
Phase 3 (Mature):    Session-aware re-ranking + contextual bandits for exploration
```

Don't over-engineer personalization at launch. Get visual search right first. Add personalization iteratively.

---

## Interview Checkpoint

1. **"How do you personalize visual search without making results feel creepy?"**
   - Use implicit signals (clicks, purchases) rather than explicit profiling. Allow users to clear history. Show diversity alongside personalized results. Transparency: "Based on your recent purchases."

2. **"How do you handle the cold-start problem for new users?"**
   - Trending/popular results, optional onboarding (select preferred categories/styles), fast warm-up from just 2–3 interactions using nearest-neighbor user matching.

3. **"How do you prevent filter bubbles in personalized visual search?"**
   - Diversity regularization in re-ranking: penalize results too similar to each other. Exploration traffic (epsilon-greedy or UCB). "Explore similar styles" UX pattern to encourage serendipity.

4. **"How do you know personalization is actually helping (not just overfitting to past behavior)?"**
   - A/B test personalized vs. non-personalized. Track not just CTR but diversity and new brand/category discovery rate. Monitor for declining diversity over time (filter bubble signal).
