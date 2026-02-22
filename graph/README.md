# People You May Know (PYMK) — ML System Design

LinkedIn-style connection recommendation system. Given a user, produce a ranked list of people they should connect with.

**Scale:** 1 billion users, ~1,000 connections per user on average (~500B edges total).

---

## Documents

| # | File | What it covers |
|---|------|----------------|
| 01 | [ML Objectives](docs/01_ml_objectives.md) | Business goals, ML framing, success metrics |
| 02 | [System I/O](docs/02_system_io.md) | API contract, data flows, latency budget |
| 03 | [Model Choices](docs/03_model_choices.md) | 5 options from heuristics to GNN to LLM, with trade-offs |
| 04 | [Data Preparation](docs/04_data_preparation.md) | Graph data, labels, sampling strategies |
| 05 | [Feature Engineering](docs/05_feature_engineering.md) | Graph, profile, behavioral, and embedding features |
| 06 | [Model Development](docs/06_model_development.md) | Training pipelines for two recommended approaches |
| 07 | [Evaluation](docs/07_evaluation.md) | Offline metrics, online A/B testing, guardrails |
| 08 | [Model Hosting](docs/08_model_hosting.md) | Serving architecture, ANN index, caching |
| 09 | [End-to-End System](docs/09_end_to_end.md) | Full pipeline walkthrough + mini Python implementation |
| 10 | [Challenges](docs/10_challenges.md) | Scale, cold start, feedback loops, privacy |
| 11 | [Summary](docs/11_summary.md) | Architecture diagram, component map, data flow |
| 12 | [Scaling](docs/12_scaling.md) | Distributed training, incremental updates, sharding |

---

## Two-Stage Pipeline (TL;DR)

```
User Query
    │
    ▼
[Stage 1: Candidate Generation]
  - 2nd/3rd-degree graph traversal  → up to 100K candidates
  - ANN search on node embeddings   →
  Reduce to ~1,000 candidates
    │
    ▼
[Stage 2: Ranking]
  - Rich feature vector per (user, candidate) pair
  - Gradient-boosted or neural ranking model
  Reduce to top 20–50 recommendations
    │
    ▼
Ranked list returned to user
```

---

## Interview Focus Areas

- Why two stages? (scale: can't score 1B candidates with a heavy model)
- How to handle cold-start users with no connections?
- How does a Graph Neural Network differ from a simple embedding lookup?
- How do you generate training labels at scale?
- How do you keep embeddings fresh without retraining daily?
- What are the top scaling bottlenecks?
