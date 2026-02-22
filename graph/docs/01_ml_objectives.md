# 01 — ML Objectives

## Business Context

LinkedIn's "People You May Know" is one of the highest-ROI features on any professional network. A user who grows their network becomes more engaged, posts more content, stays on the platform longer, and is more likely to convert to a premium subscription.

The core goal is: **help users discover relevant connections they would not have found on their own**.

---

## Clarifying Questions (ask these in an interview)

Before designing anything, scope the problem with these probes:

| Question | Why it matters |
|---|---|
| Is this for all users or a specific segment? | Cold-start users need different treatment |
| Are we optimizing for connection requests sent or accepted? | "Sent" inflates volume; "accepted" measures real value |
| How many recommendations per user per day? | Drives latency and throughput budgets |
| What's the acceptable latency? | Determines online vs. precomputed trade-off |
| Can we show the same person twice across sessions? | Affects deduplication and diversity logic |
| Are there privacy restrictions (e.g., private accounts)? | Affects candidate pool construction |

---

## Business Objective → ML Objective

```
Business Goal                       ML Formulation
─────────────────────────────────   ──────────────────────────────────────────
Grow user connections               Predict P(user_A accepts connection to user_B)
Increase engagement                 Maximize expected accepted connections per session
Retain users                        Ensure recommendations feel relevant and non-spammy
```

**Primary ML Task:** Pointwise or pairwise ranking — given a query user and a candidate user, predict the probability that the query user will send (and the candidate will accept) a connection request.

---

## Success Metrics

### Primary (online)
| Metric | Target | Notes |
|---|---|---|
| Connection Accept Rate | > baseline | % of recommended connections that are accepted |
| 7-day New Connection Rate | ↑ vs. control | New connections formed within 7 days of seeing recommendation |
| Weekly Active Users (WAU) | ↑ vs. control | Feature engagement proxy |

### Secondary (quality guardrails)
| Metric | Notes |
|---|---|
| Dismiss / "Not Interested" Rate | Must not increase — signals spam |
| Block / Report Rate | Hard guardrail: any increase kills the launch |
| Recommendation Diversity | Avoid echo chambers (same company, same school only) |

### Offline (model development proxy)
| Metric | Notes |
|---|---|
| AUC-ROC | Binary label: accepted = 1, ignored/dismissed = 0 |
| Precision@K / Recall@K | K = number of slots shown to user |
| NDCG@K | Ranking quality when scores are available |

---

## Constraints

| Constraint | Value | Implication |
|---|---|---|
| Total users | 1 billion | Cannot score all candidates per query |
| Avg connections per user | 1,000 | Graph has ~500B edges total |
| 2nd-degree candidates | up to ~1M per user | Must prune aggressively in Stage 1 |
| Latency budget | < 150ms P99 | Requires precomputed embeddings + ANN index |
| Daily active users | ~300M (estimate) | Drives throughput requirements |
| Recommendation freshness | Within 24h of new connection | Affects update frequency |

---

## What "Good" Looks Like

A good recommendation:
- The two users share meaningful common ground (mutual connections, same industry, same company, co-attended same school)
- At least one of them has shown implicit interest (viewed the other's profile, engaged with their content)
- The recommendation feels organic, not obviously algorithmic

A bad recommendation:
- People already connected (duplicate work)
- People who have previously dismissed each other
- People with no obvious relationship signal (random strangers)
- Celebrities or very high-degree nodes dominating all recommendations

---

## Interview Checkpoint

**Q: How do you translate "grow the network" into a training label?**

The clean label is a **positive interaction**: user A accepted a connection from user B, or user A sent a connection to user B who accepted. Negatives are harder — a non-event (no connection formed) does not mean the recommendation was bad. Better negatives: explicit "dismiss" actions, or impressions-without-clicks after multiple sessions. This label quality problem is a key interview talking point.

**Q: Why not just optimize for "connection requests sent" instead of "accepted"?**

Sending requests is easy to game and inflates spam. If the model learns that aggressive recommendations lead to more sends, it will recommend inappropriately and degrade user trust. Accepted connections are a much stronger signal of true value.
