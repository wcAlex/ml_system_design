# 02 — System Input & Output

## API Contract

### Online (real-time) endpoint

```
GET /pymk/recommendations?user_id={uid}&limit=20&context={...}

Response:
{
  "user_id": "u_123",
  "recommendations": [
    {
      "candidate_id": "u_456",
      "score": 0.92,
      "reason_codes": ["mutual_connections:12", "same_company"],
      "mutual_connection_count": 12
    },
    ...
  ],
  "request_id": "req_abc",
  "served_at": "2024-01-15T10:23:00Z"
}
```

### Inputs

| Field | Type | Description |
|---|---|---|
| `user_id` | string | Query user identifier |
| `limit` | int | Number of recommendations to return (default 20) |
| `context.device` | enum | mobile / desktop — affects diversity/count |
| `context.session_id` | string | For deduplication across refreshes |
| `context.location` | string | Optional — for location-based boosting |

### Outputs

| Field | Description |
|---|---|
| `candidate_id` | Recommended user's ID |
| `score` | Ranking score (0–1), used for ordering |
| `reason_codes` | Human-readable reasons (shown to user as "X mutual connections") |
| `mutual_connection_count` | Pre-computed, shown in UI |

---

## Data Flows

```
                      ┌──────────────────────────────────────┐
                      │         OFFLINE PIPELINE              │
                      │  (runs nightly / every few hours)     │
  Graph DB ──────────►│  1. Graph traversal → candidate sets  │
  Profile DB ─────────►  2. Node embedding training (GNN)     │
  Interaction Logs ───►  3. ANN index build                   │
                      │  4. Feature store update              │
                      └──────────────────┬───────────────────┘
                                         │ precomputed data
                                         ▼
                      ┌──────────────────────────────────────┐
                      │           FEATURE STORE               │
                      │  - node embeddings (user → vector)   │
                      │  - precomputed graph features         │
                      │  - candidate lists per user           │
                      └──────────────────┬───────────────────┘
                                         │
                      ┌──────────────────▼───────────────────┐
                      │        ONLINE SERVING PIPELINE        │
User request ────────►│  1. Fetch precomputed candidates      │
                      │  2. Retrieve features from store      │
                      │  3. Run ranking model (lightweight)   │
                      │  4. Apply business filters            │
                      │  5. Return top-K                      │
                      └──────────────────────────────────────┘
```

---

## Latency Budget Breakdown

Total P99 budget: **150ms**

| Step | Budget | Notes |
|---|---|---|
| Network + routing | 10ms | CDN/load balancer |
| Candidate fetch (precomputed) | 5ms | Key-value store lookup |
| Feature retrieval | 20ms | Parallel fetch from feature store |
| Ranking model inference | 30ms | ~1000 candidates × lightweight model |
| Business filters + dedup | 5ms | Block list, already-connected, etc. |
| Serialization + response | 5ms | JSON encoding |
| **Total** | **75ms** | ~50% headroom for P99 spikes |

**Key insight:** The heavy computation (graph traversal, GNN inference) happens **offline**. The online path only fetches precomputed data and runs a lightweight ranker. This is the only way to serve 1B users at < 150ms.

---

## Precomputed vs. Real-Time Trade-offs

| Approach | Latency | Freshness | Complexity |
|---|---|---|---|
| Fully precomputed (recommended) | Very low | Hours (stale) | Moderate |
| Hybrid: precomputed candidates + online ranking | Low | Minutes for ranking signals | Higher |
| Fully real-time graph traversal | High (>1s) | Real-time | Very high |

**Recommended:** Precomputed candidates with online ranking. Candidate lists refresh every 1–4 hours; ranking incorporates recent signals (last session activity) fetched at serving time.

---

## Key Design Decisions

**1. How many candidates to generate in Stage 1?**
- Too few: miss good recommendations
- Too many: ranking is slow
- Sweet spot: ~500–2,000 candidates per user

**2. How to handle new connections?**
- When user A connects with user B, immediately remove B from A's recommendation list
- This requires a fast invalidation path (update key-value store within seconds)

**3. What about real-time profile views?**
- If user A just viewed user B's profile, user B should appear higher in PYMK
- Inject this "hot signal" at serving time — add a small score boost without recomputing the full ranking

---

## Interview Checkpoint

**Q: Why precompute candidates instead of doing it online?**

At 1B users, even a simple 2nd-degree graph traversal for a single user can touch up to 1M nodes. Doing this in real time for every page load is infeasible. Precomputation amortizes the expensive graph work across all users, done in batch.

**Q: How do you keep recommendations fresh if candidates are precomputed?**

Run the candidate generation pipeline incrementally — when a user forms a new connection, trigger a partial refresh of their candidate set. The ranking step at serving time can also incorporate real-time signals (recent profile views, recent activity) to re-order stale candidates effectively.
