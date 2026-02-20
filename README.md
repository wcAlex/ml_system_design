# ML System Design

Implementation and in-depth study of production-grade ML systems, with a focus on senior-level machine learning engineering and system design interviews.

Each project is built from first principles, covering the full lifecycle from problem framing through model development, serving infrastructure, and scaling. Design decisions are explained with explicit trade-offs, and real-world references from Google, YouTube, Meta, and Amazon are prioritized where applicable.

---

## Goals

- Understand the **end-to-end architecture** of popular ML systems at scale
- Build intuition for **model selection trade-offs** across retrieval, ranking, and re-ranking stages
- Practice the structured thinking required in **senior ML system design interviews**
- Cover the most common interview pressure points: bottlenecks, cold start, data sparsity, latency vs. accuracy, and scaling strategies

---

## Repository Structure

```
ml_system_design/
├── recommendation/
│   └── video_recommendation/       # YouTube-style video recommendation system
├── search/
│   └── image_search/               # Image similarity search system
├── ml_algorithms/                  # Core algorithm deep-dives with demos
│   └── nearest_neighbour/          # KNN and ANN (HNSW, FAISS, ScaNN)
├── feature_engineering/            # Common feature engineering techniques
└── graph/                          # Graph-based ML (in progress)
```

---

## Projects

### 1. Video Recommendation System
> `recommendation/video_recommendation/`

A YouTube-style recommendation system designed to maximize user engagement. Covers the full candidate retrieval → ranking → re-ranking pipeline.

| Doc | Topic |
|-----|-------|
| `01_probe_questions.md` | Clarifying questions and problem scoping |
| `02_scoring_and_guardrails.md` | Objective definition and guardrails |
| `03_recommendation_landscape.md` | Industry landscape and system overview |
| `04_data_collection.md` | Data sources, labeling strategies, feedback signals |
| `05_serving_system.md` | Online serving architecture |
| `06_ranking_deep_dive.md` | Ranking model design (DCN, DIN, DHEN) |
| `07_reranking_deep_dive.md` | Re-ranking: diversity, freshness, policy |
| `08_two_tower_training_deep_dive.md` | Two-tower retrieval model training |
| `09_inference_control_plane.md` | Inference infrastructure and control plane |

**Key models:** Two-tower retrieval, Matrix Factorization, Multi-task ranking, MMR re-ranking

---

### 2. Image Search System
> `search/image_search/`

A content-based image retrieval (CBIR) system that retrieves and ranks images by visual similarity to a query image.

| Doc | Topic |
|-----|-------|
| `01_ml_objectives.md` | ML objectives and success metrics |
| `02_system_io.md` | System inputs, outputs, and constraints |
| `03_model_choices.md` | Embedding model options and trade-offs |
| `04_data_preparation.md` | Dataset construction and preprocessing |
| `05_feature_engineering.md` | Visual feature extraction strategies |
| `06_model_development.md` | Model training approaches (contrastive, supervised) |
| `07_evaluation.md` | Offline and online evaluation metrics |
| `08_model_hosting.md` | Serving infrastructure and latency optimization |
| `09_end_to_end.md` | End-to-end system walkthrough |
| `10_challenges.md` | Hard problems and common failure modes |
| `11_summary.md` | Architecture summary and data flow |
| `12_scaling.md` | Scaling: model size, index size, query throughput |
| `13_personalization.md` | Personalization layer on top of retrieval |

**Key models:** CLIP, ResNet/ViT embeddings, FAISS/ScaNN ANN index, contrastive learning

---

## Core Algorithms

> `ml_algorithms/`

Standalone deep-dives into foundational algorithms used across the projects above.

| Algorithm | Notes |
|-----------|-------|
| K-Nearest Neighbors (KNN) | Exact search, distance metrics, complexity analysis |
| Approximate Nearest Neighbors (ANN) | HNSW, FAISS, ScaNN — trade-offs and benchmarks |
| Gradient Boosted Trees | XGBoost/LightGBM for ranking and scoring |
| Mixture of Experts (MoE) | Architecture, routing, and scaling behavior |

Each includes an interactive Jupyter notebook demo.

---

## Interview Focus Areas

Each project is structured to address the most common senior-level ML system design interview topics:

- **Retrieval vs. ranking vs. re-ranking** — why you need each stage and how they interact
- **Cold start** — handling new users and new items with no history
- **Scalability** — serving billions of items with sub-100ms latency
- **Offline vs. online evaluation** — metrics that matter and how to connect them to business goals
- **Feature engineering** — what signals to use and how to compute them at scale
- **Model iteration** — how to safely roll out new models and measure improvement
- **Failure modes** — filter bubbles, popularity bias, position bias, data leakage

---

## Each Project Covers

1. ML objectives and business alignment
2. System inputs and outputs
3. Model choices with trade-offs (2-3 options per decision)
4. Data preparation and pipeline design
5. Feature engineering
6. Model development (at least two approaches)
7. Offline and online evaluation
8. Model hosting and inference
9. End-to-end mini implementation
10. Challenges and failure modes
11. Architecture summary
12. Scaling strategies

---

## Status

| Project | Status |
|---------|--------|
| Video Recommendation | In progress |
| Image Search | In progress |
| Video Search | Planned |
| Harmful Content Detection | Planned |
| Graph-based Recommendation | Planned |
