# 08 — Model Hosting & Inference

## The Production Challenge

Training a great model is half the battle. Serving it at low latency, high throughput, and high availability — on a 10M+ product index — is the other half.

---

## Two-Phase Serving Architecture

```
Phase 1: RETRIEVAL          Phase 2: RE-RANKING
──────────────────          ─────────────────────
Query image                 500 ANN candidates
    │                            │
    ▼ ~20–40ms                  ▼ ~30–50ms
Embedding Model             Lightweight Ranker
(GPU inference)             (attribute match,
    │                        personalization,
    ▼                        business rules)
512-d vector                     │
    │                            ▼
    ▼ ~20–40ms              Top-20 results
ANN Index Search            returned to user
(FAISS/ScaNN)
    │
    ▼
500 candidates
```

---

## Embedding Service (Query-Time)

### Deployment Options

#### Option 1: GPU Inference Server (Recommended for Production)
```
Framework: TorchServe / Triton Inference Server
Hardware: NVIDIA T4 (cost-efficient) or A10G (performance)
Batch size: 1 (online, latency-focused)
Model format: TorchScript or ONNX (faster than eager mode)

Throughput: ~1000 QPS per GPU (CLIP ViT-B/16)
Latency: ~15–25ms per request (GPU inference)
```

**Why ONNX?** Converts model to optimized runtime; 1.5–2x speedup over PyTorch eager mode.

```python
# Export CLIP image encoder to ONNX
import torch
import clip

model, preprocess = clip.load("ViT-B/16")
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model.encode_image,
    dummy_input,
    "clip_image_encoder.onnx",
    input_names=["image"],
    output_names=["embedding"],
    dynamic_axes={"image": {0: "batch_size"}}
)
```

#### Option 2: CPU Inference with Quantization
- INT8 quantization: 4x memory reduction, 2–3x speedup on CPU
- Acceptable for smaller models (ResNet-50) or cost-constrained environments
- ViT models degrade more with quantization than CNNs

```python
# Dynamic quantization (post-training, no calibration data needed)
import torch.quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

#### Option 3: Distilled Model (Small + Fast)
- Train a smaller student model (ResNet-50) to mimic CLIP ViT-L embeddings
- Knowledge distillation: `L = α·task_loss + (1-α)·KL(student, teacher)`
- 4x smaller, 4x faster — quality drops ~5–10% on Recall@10
- Pinterest and Google both use distillation for their mobile/edge inference

---

## ANN (Approximate Nearest Neighbor) Index

This is the core retrieval engine. Finds the top-K most similar embeddings from 10M+ stored products.

### Why Approximate?
- Exact kNN on 10M × 512-d floats: ~20 billion operations per query — too slow
- ANN trades tiny quality loss (~1–2% recall) for 100–1000x speedup

### Option 1: FAISS (Facebook AI Similarity Search)
```
Library: faiss (Meta, open source)
Index types:
  - IndexFlatL2:    exact search, no compression (baseline)
  - IndexIVFFlat:   inverted file index (clustering) — fast
  - IndexIVFPQ:     product quantization (compressed) — memory-efficient
  - IndexHNSWFlat:  hierarchical navigable small world — low latency

For 10M products, 512-d:
  - IndexIVFPQ: ~1–2GB RAM, ~30ms P99
  - IndexHNSWFlat: ~20GB RAM, ~10ms P99 (better latency, more memory)
```

```python
import faiss
import numpy as np

# Build FAISS IVF-PQ index
d = 512          # embedding dimension
nlist = 1000     # number of voronoi cells (clusters)
m = 32           # number of sub-quantizers (PQ compression)

quantizer = faiss.IndexFlatIP(d)  # Inner product (for L2-normalized vectors = cosine)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)

# Train on representative sample
index.train(embeddings_sample)

# Add all product embeddings
index.add(product_embeddings)

# Search: find top-100 for a query
index.nprobe = 50  # check 50 clusters (trade-off: more = better recall, slower)
distances, indices = index.search(query_embedding.reshape(1, -1), k=500)
```

**Pros:** Battle-tested, fast, flexible
**Cons:** Single-machine by default (sharding needed at scale), no native persistence management

### Option 2: ScaNN (Google)
```
Developed by Google; used in Google Search and YouTube
Key innovation: anisotropic vector quantization (better than PQ for inner product)
Performance: 2-5x faster than FAISS at same recall on standard benchmarks
Language: C++ with Python bindings
```

```python
import scann

searcher = scann.scann_ops_pybind.builder(
    product_embeddings, num_neighbors=500, distance_measure="dot_product"
).tree(
    num_leaves=2000,
    num_leaves_to_search=100,
    training_sample_size=250000
).score_ah(
    dimensions_per_block=2,
    anisotropic_quantization_threshold=0.2
).reorder(500).build()

neighbors, distances = searcher.search(query_embedding)
```

**Pros:** Fastest known ANN for inner product; used in production at Google scale
**Cons:** Less flexible than FAISS, harder to customize

### Option 3: Managed Vector Database
| Product | Company | Notes |
|---|---|---|
| **Pinecone** | Pinecone | Fully managed, simple API, pay-per-use |
| **Weaviate** | Open source | Graph + vector search, hybrid search |
| **Milvus** | Zilliz | Open source, distributed, enterprise |
| **Qdrant** | Open source | Rust-based, fast, good for filtering |
| **pgvector** | PostgreSQL | Simple: vector search in Postgres |

**When to use managed:** Small team, don't want to manage infrastructure. Trade-off: cost at scale, less control over index tuning.

**When to use FAISS/ScaNN:** Large scale (>100M), cost-sensitive, need fine-grained tuning.

### ANN Index Sharding (Scale)

At 100M products, a single FAISS index may not fit in RAM:

```
Product catalog (100M items)
         │
    ┌────┴────┐
    │  Hash   │ (by product_id mod N)
    └────┬────┘
    │         │
  Shard 1   Shard 2  ...  Shard N  (each handles 10M items)
    │         │
    └────┬────┘
         │ Merge top-K results
    Global re-ranker
```

All shards searched in parallel; results merged and re-ranked.

---

## Offline Indexing Pipeline

### Product Embedding Pipeline

```
New/Updated Products (from catalog DB)
         │
         ▼ Kafka / SQS stream
┌─────────────────────────────────┐
│  Embedding Batch Job (Spark)    │
│  - GPU batch inference          │
│  - Batch size: 512 images/step  │
│  - Throughput: ~50K products/hr │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Embedding Store (Parquet/S3)   │
│  + Feature Store (Redis)        │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  ANN Index Builder              │
│  - Rebuild full index: nightly  │
│  - Incremental updates: hourly  │
└─────────────┬───────────────────┘
              │
              ▼
         Serving Index
    (read by retrieval service)
```

### Index Update Strategies

| Strategy | Freshness | Complexity | Cost |
|---|---|---|---|
| **Full rebuild nightly** | 24 hours | Low | High compute nightly |
| **Incremental hourly** | 1 hour | Medium | Streaming embedding + partial index update |
| **Real-time streaming** | Minutes | High | Kafka + online index updates (Milvus supports this) |

**Recommendation:** Full rebuild nightly + incremental hourly for priority products (new inventory, trending items).

---

## Caching Strategy

| Cache Level | What | TTL | Notes |
|---|---|---|---|
| **CDN / Gateway** | Responses for identical query images | 1 hour | Hash image → cache key; rare for unique photos |
| **Embedding cache** | Embeddings for common query images | 24 hours | Stock photos, product page images |
| **ANN result cache** | Top-500 for common embeddings | 1 hour | For high-traffic identical queries |

**Warning:** Caching reduces freshness. Don't cache personalized results.

---

## Auto-Scaling

```
Load Balancer
     │
┌────┼────┐
│    │    │
Emb  Emb  Emb  ← Auto-scale: scale out GPU pods based on QPS
Svc  Svc  Svc    Target: avg GPU utilization < 70%
└────┼────┘
     │
┌────┼────┐
│    │    │
ANN  ANN  ANN  ← Scale out based on search latency P99
Idx  Idx  Idx
```

- GPU instances are expensive — use spot instances for non-serving workloads (embedding batch jobs)
- ANN search is CPU-bound — use CPU-optimized instances for the retrieval service

---

## Interview Checkpoint

1. **"How do you serve 10K QPS with <200ms latency?"**
   - Horizontal scaling: multiple GPU pods for embedding, multiple ANN shards searched in parallel. Cache common queries. Use efficient model format (ONNX, TorchScript).

2. **"FAISS vs. managed vector DB — when do you choose each?"**
   - FAISS: large scale, custom tuning, cost-sensitive. Managed (Pinecone, Milvus): faster to launch, easier ops, good for <50M products. The break-even point depends on team size and scale.

3. **"How do you update the index without downtime?"**
   - Blue-green deployment: build new index in parallel, health-check, switch traffic. FAISS supports read-only index serving — write new index to disk, reload atomically.

4. **"What happens to the index when you retrain the model?"**
   - All product embeddings must be regenerated (since embedding space changed). This is a full re-indexing job. Schedule nightly or on model update. Use embedding versioning — store embeddings with model version tag.
