# 05 — Feature Engineering

## Feature Categories

There are four categories of features for PYMK. The best models use all four.

```
1. Graph Features       — structural signals from the connection graph
2. Profile Features     — who the two users are (industry, title, school)
3. Behavioral Features  — interaction history between the two users
4. Embedding Features   — dense representations from GNN or embedding model
```

All features are computed for a **(query_user, candidate_user)** pair.

---

## 1. Graph Features

These are the most important features for PYMK. They capture the structural relationship between the two users.

### Mutual Connection Features

| Feature | Formula | Intuition |
|---|---|---|
| `common_neighbor_count` | `\|N(A) ∩ N(B)\|` | Raw count of shared connections |
| `jaccard_similarity` | `\|N(A) ∩ N(B)\| / \|N(A) ∪ N(B)\|` | Normalized mutual connections |
| `adamic_adar_score` | `Σ 1/log(\|N(z)\|)` for z in common neighbors | Weights rare mutual friends higher |
| `preferential_attachment` | `\|N(A)\| × \|N(B)\|` | Higher-degree nodes more likely to connect |

### Network Distance Features

| Feature | Description |
|---|---|
| `shortest_path_length` | Is the candidate 2nd, 3rd, or 4th degree? |
| `num_2nd_degree_paths` | Number of distinct shortest paths of length 2 |
| `graph_distance_bucket` | Discretized: {same_connection=1, 2nd_degree=2, 3rd_degree=3, farther=4} |

### Community / Cluster Features

| Feature | Description |
|---|---|
| `same_community` | Do they belong to the same detected graph community? |
| `community_overlap_score` | Fraction of shared community memberships |
| `clustering_coeff_query` | How tightly clustered is the query user's neighborhood? |

### Degree Features

| Feature | Description |
|---|---|
| `query_user_degree` | Number of connections the query user has |
| `candidate_degree` | Number of connections the candidate has |
| `degree_ratio` | candidate_degree / query_user_degree |

**Watch out:** High-degree nodes (influencers, hubs) will have high Adamic-Adar and common neighbor scores with everyone. Consider log-normalizing degree features and filtering out the top 0.1% high-degree nodes from candidate sets.

---

## 2. Profile Features

These capture similarity in professional background.

### Identity Features

| Feature | Type | Example |
|---|---|---|
| `same_company_current` | binary | Both work at Google now |
| `same_company_past` | binary | Both worked at Facebook (different times) |
| `company_overlap_count` | int | Number of companies in common |
| `same_school` | binary | Both attended MIT |
| `same_graduation_year` | binary | Graduated same year |
| `education_overlap_count` | int | Number of schools in common |

### Professional Similarity

| Feature | Type | Notes |
|---|---|---|
| `same_industry` | binary | Both in "Technology" |
| `industry_cosine_sim` | float | Industry embedding similarity |
| `title_cosine_sim` | float | Job title embedding similarity (BERT) |
| `seniority_diff` | int | |IC vs. Manager| difference |
| `skill_overlap_count` | int | Number of skills in common |
| `skill_jaccard_sim` | float | Normalized skill overlap |

### Location Features

| Feature | Notes |
|---|---|
| `same_city` | Strong signal for local networking events |
| `same_country` | Weaker signal |
| `geo_distance_km` | Continuous proximity |

### Profile Quality

| Feature | Notes |
|---|---|
| `candidate_profile_completeness` | Incomplete profiles = lower quality recommendation |
| `candidate_account_age_days` | New accounts may be bots; old accounts more trustworthy |
| `query_user_activity_level` | Active users → better signal; dormant users → noisy |

---

## 3. Behavioral Features

These capture any prior interaction between the two users.

| Feature | Signal |
|---|---|
| `query_viewed_candidate_profile` | binary: has A viewed B's profile? |
| `candidate_viewed_query_profile` | binary: has B viewed A's profile? |
| `mutual_profile_views` | Both viewed each other |
| `query_profile_view_recency_days` | How recently did A view B? |
| `co_engaged_with_content` | Both liked/commented on the same post |
| `query_messaged_candidate` | Even one message is a very strong signal |
| `co_attended_event` | Both RSVPed/attended the same event |
| `candidate_endorsed_query_skill` | B endorsed one of A's skills |
| `search_click` | A searched for and clicked on B's profile |

**Key insight:** Even one profile view or content co-engagement is a much stronger signal than dozens of common connections. These behavioral features often dominate the ranking model.

---

## 4. Embedding Features

Dense vector features that capture complex patterns not expressible as hand-engineered features.

### Node Embeddings (from GNN or Node2Vec)

```
query_embedding:     (128-d float vector) — query user's GNN representation
candidate_embedding: (128-d float vector) — candidate's GNN representation
embedding_dot_product:  scalar            — similarity score
embedding_cosine_sim:   scalar            — normalized similarity
```

**How to use them in a downstream model:**
- Option A: Use dot product / cosine similarity as a single feature
- Option B: Concatenate both embeddings and let the ranker learn the interaction (more flexible but higher-dimensional input)
- Option C: Element-wise product — captures dimension-specific similarity

### Profile Text Embeddings (from LLM/BERT)

```
query_profile_text_embedding:     (768-d) — encoded job title + bio + skills
candidate_profile_text_embedding: (768-d)
profile_text_cosine_sim:           scalar
```

Captures semantic similarity: "Machine Learning Engineer" and "Deep Learning Scientist" are semantically close even though no keywords match.

---

## Feature Engineering Pipeline

```
Offline (daily batch):
  1. Graph traversal → compute graph features for all (user, candidate) pairs
  2. Profile processing → normalize, encode categorical features
  3. GNN training → produce node embeddings
  4. Write features to Feature Store

Online (at serving):
  1. Fetch precomputed features from Feature Store
  2. Compute recency-dependent features (e.g., last_viewed_hours_ago)
  3. Assemble feature vector and pass to ranker
```

---

## Feature Importance (Rough Ranking)

Based on published research from LinkedIn and similar systems:

```
Tier 1 (most important):
  - common_neighbor_count / adamic_adar_score
  - query_viewed_candidate_profile (recency-weighted)
  - same_company_current / same_school

Tier 2 (high value):
  - embedding_cosine_sim (GNN)
  - skill_overlap_count
  - co_engaged_with_content

Tier 3 (helpful signals):
  - location proximity
  - seniority_diff
  - candidate_profile_completeness
  - degree features (log-normalized)
```

---

## Interview Checkpoint

**Q: How do you prevent the "celebrity hub" problem where high-degree nodes dominate recommendations?**

Three techniques:
1. Use **Adamic-Adar** instead of raw common neighbor count — it penalizes high-degree mutual friends.
2. **Log-normalize** degree features so outliers (degree=10,000) don't overwhelm typical users (degree=100).
3. **Filter high-degree nodes** (top 0.01% by connection count) from the candidate set entirely, or cap their contribution to scoring.

**Q: If the GNN already captures graph structure, why do you still need hand-engineered graph features?**

The GNN embedding is a dense summarization of the local graph structure. It captures complex patterns but in an opaque way. Hand-engineered features like `common_neighbor_count` and `same_company` are:
1. **Interpretable** — you can show the user "12 mutual connections" as a reason
2. **Fast** — no GNN inference needed at serving time if precomputed
3. **Robust** — they don't go stale if the GNN model hasn't been retrained recently

The best production systems use both: GNN embeddings for deep pattern capture + hand-engineered features for explainability and stability.
