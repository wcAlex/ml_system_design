# Image Search System — E-commerce Visual Product Search

A production-grade ML system design for visual product search (à la Amazon StyleSnap, Pinterest Lens, Google Shopping Lens). Users upload a photo and get back ranked, shoppable results.

---

## Project Structure

```
image_search/
├── docs/
│   ├── 01_ml_objectives.md        # Business + ML objectives, metrics
│   ├── 02_system_io.md            # Inputs, outputs, API contracts
│   ├── 03_model_choices.md        # CNN vs ViT vs CLIP vs DINO — with trade-offs
│   ├── 04_data_preparation.md     # Data sources, labeling, augmentation
│   ├── 05_feature_engineering.md  # Embeddings, attributes, multi-scale
│   ├── 06_model_development.md    # Training approaches — metric learning, CLIP fine-tuning
│   ├── 07_evaluation.md           # Offline/online metrics, A/B testing
│   ├── 08_model_hosting.md        # Serving architecture, ANN indexes
│   ├── 09_end_to_end.md           # Full system design + mini implementation guide
│   ├── 10_challenges.md           # Failure modes, bottlenecks, interview talking points
│   ├── 11_summary.md              # Architecture diagram, data flow, component map
│   ├── 12_scaling.md              # Scaling model size, index, data pipelines
│   └── 13_personalization.md      # User history, re-ranking, contextual bandits
├── src/
│   ├── indexer.py                 # Product catalog embedding + FAISS index
│   ├── searcher.py                # Query embedding + ANN retrieval
│   ├── reranker.py                # Attribute-based + personalized re-ranking
│   └── demo.py                   # End-to-end demo
└── requirements.txt
```

## The Core Idea

> Given a query image (e.g., a sneaker photo), return the top-K visually similar products from a catalog of millions of items — in under 200ms.

**Three-stage pipeline:**
1. **Embedding** — encode query image into a dense vector
2. **ANN Retrieval** — find approximate nearest neighbors in the product index (~millions of items)
3. **Re-ranking** — apply business rules, attribute filters, and personalization

## Quick Start

Read the docs in order. Each doc ends with **Interview Checkpoint** questions.

---

## Key Industry References
- Pinterest: [Visual Search at Pinterest (2017)](https://arxiv.org/abs/1706.04069)
- Amazon: StyleSnap — multi-modal product matching
- Google: Lens uses CLIP-family models + Shopping Graph
- Meta: Image similarity search with FAISS
