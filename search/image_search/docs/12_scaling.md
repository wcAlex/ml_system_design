# 12 — Scaling

## Scaling Is Not Just "Add More Machines"

True scaling means making deliberate trade-offs between cost, latency, throughput, and quality. This section covers scaling in three dimensions:
1. **Model size scaling** (larger → better quality, but slower)
2. **Inference/serving scaling** (handling 10K+ QPS)
3. **Data pipeline scaling** (100M+ products, billions of events)

---

## Dimension 1: Model Size Scaling

### The Scaling Ladder

```
Stage 1: ResNet-50                  25M params    Recall@10: ~70%    Latency: ~5ms
Stage 2: CLIP ViT-B/32             151M params    Recall@10: ~78%    Latency: ~10ms
Stage 3: CLIP ViT-B/16             150M params    Recall@10: ~84%    Latency: ~20ms
Stage 4: CLIP ViT-L/14             428M params    Recall@10: ~88%    Latency: ~45ms
Stage 5: CLIP ViT-L/14 @336px      428M params    Recall@10: ~90%    Latency: ~80ms
```

**Key insight:** Recall@10 improves logarithmically with model size. Going from ViT-B to ViT-L gives ~4% gain but 2x latency cost. The business must decide if that trade-off is worth it.

### Model Compression for Scaling Without Latency Cost

#### Knowledge Distillation
Train a smaller student model to replicate a large teacher model:

```python
class DistillationTrainer:
    """
    Train student model (ViT-S) to mimic teacher (ViT-L) embeddings.
    """
    def __init__(self, teacher, student, temperature=4.0, alpha=0.7):
        self.teacher = teacher  # frozen
        self.student = student  # trainable
        self.temperature = temperature
        self.alpha = alpha

    def distillation_loss(self, images, labels=None):
        with torch.no_grad():
            teacher_embed = self.teacher.encode_image(images)
            teacher_embed = F.normalize(teacher_embed, dim=-1)

        student_embed = self.student.encode_image(images)
        student_embed = F.normalize(student_embed, dim=-1)

        # Embedding-level distillation: student should match teacher embeddings
        distill_loss = 1 - F.cosine_similarity(student_embed, teacher_embed).mean()

        # Optional: task loss (contrastive on student)
        task_loss = info_nce_loss(student_embed, positives)

        return self.alpha * distill_loss + (1 - self.alpha) * task_loss
```

**Results:** ViT-S distilled from ViT-L ≈ 85% of teacher's quality at 5x lower latency. Pinterest and Google use this pattern extensively.

#### Quantization

```python
# Post-training quantization (no fine-tuning needed)
import torch.quantization

# INT8 dynamic quantization: 4x smaller, ~2x faster on CPU
quantized = torch.quantization.quantize_dynamic(
    model.cpu(),
    {torch.nn.Linear, torch.nn.MultiheadAttention},
    dtype=torch.qint8
)

# On GPU: use TensorRT for INT8/FP16 quantization
# FP16: 2x memory savings, ~1.5x speedup, minimal quality loss
# INT8: 4x memory savings, ~2-3x speedup, ~1-2% quality loss
```

| Method | Memory | Speedup | Quality Loss |
|---|---|---|---|
| FP32 (baseline) | 1x | 1x | 0% |
| FP16 | 0.5x | 1.5x | <0.5% |
| INT8 (PTQ) | 0.25x | 2-3x | 1-2% |
| INT8 (QAT) | 0.25x | 2-3x | <1% |

#### Embedding Dimension Reduction (PCA)

```python
from sklearn.decomposition import PCA

# Reduce 512-d embeddings to 128-d after training
pca = PCA(n_components=128)
pca.fit(training_embeddings)

# Apply to all product embeddings
compressed_embeddings = pca.transform(product_embeddings)
# Then L2-normalize
compressed_embeddings /= np.linalg.norm(compressed_embeddings, axis=1, keepdims=True)

# Index only 128-d vectors: 4x smaller index, 4x faster ANN search
# Quality loss: ~2-4% Recall@10
```

---

## Dimension 2: Inference/Serving Scaling

### QPS Scaling (10K+ Requests Per Second)

#### Horizontal Scaling

```
Load Balancer (L4/L7)
       │
  ┌────┼────┐
  │    │    │
GPU  GPU  GPU   ← N GPU pods (each handles ~1000 QPS for CLIP ViT-B/16)
Pod  Pod  Pod     Auto-scale based on GPU utilization (target: 70%)
  └────┼────┘
       │
   Embedding vectors
       │
  ┌────┼────┐
  │    │    │
ANN  ANN  ANN   ← M ANN pods (CPU-optimized, each handles ~3000 QPS)
Shard Shard Shard  Search shards in parallel, merge results
```

**Capacity math (ViT-B/16 on T4 GPU):**
- 1 T4 GPU: ~1000 QPS embedding throughput
- At 10K QPS peak: need 10 GPUs minimum (add 2x headroom for spikes → 20 GPUs)
- Cost: ~$0.50/hr per T4 spot instance → $10/hr at 20x → $7,200/month for embedding alone

#### Request Batching

```python
class BatchingEmbeddingService:
    """
    Collect queries for N ms, then process as a batch.
    Increases GPU utilization (batch size >> 1).
    """
    def __init__(self, model, max_batch_size=32, max_wait_ms=5):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = asyncio.Queue()

    async def embed_batch(self, batch_images):
        tensor = torch.stack([self.preprocess(img) for img in batch_images]).to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode_image(tensor)
            embeddings = F.normalize(embeddings, dim=-1)
        return embeddings.cpu().numpy()

    async def process(self):
        while True:
            batch = []
            # Collect up to max_batch_size requests or wait max_wait_ms
            deadline = asyncio.get_event_loop().time() + self.max_wait_ms / 1000
            while len(batch) < self.max_batch_size:
                try:
                    timeout = deadline - asyncio.get_event_loop().time()
                    if timeout <= 0:
                        break
                    item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            if batch:
                images = [item["image"] for item in batch]
                embeddings = await self.embed_batch(images)
                for item, emb in zip(batch, embeddings):
                    item["future"].set_result(emb)
```

**Result:** Batching increases GPU utilization from ~30% (single requests) to ~85%, effectively 3x throughput at same cost.

### ANN Index Scaling

#### Sharding Strategy at 100M Products

```
Product catalog: 100M items × 512-d float32 = 200GB raw embeddings
FAISS IVFFlat: ~200GB index (in RAM) → too large for one machine

Solution: Shard by product category or hash
  - 10 shards × 10M products each
  - Each shard: ~20GB RAM (fits on a 32GB machine)
  - All shards searched in parallel
  - Results merged and top-K returned

Shard by category (semantic sharding):
  Pros: Smaller search space per shard (if category is pre-filtered)
  Cons: Imbalanced shards (shoes >> pottery)

Shard by hash (uniform sharding):
  Pros: Balanced load
  Cons: Must search all shards for every query
```

#### Index Performance Tuning

```python
# FAISS IVFFlat tuning for 10M vectors
nlist = 10000    # number of Voronoi cells (clusters)
nprobe = 100     # cells to search at query time

# Trade-off:
# nprobe=10:  ~5ms,  Recall@10=75%
# nprobe=50:  ~15ms, Recall@10=88%
# nprobe=100: ~25ms, Recall@10=93%
# nprobe=500: ~80ms, Recall@10=98%

# Choose nprobe to hit your latency budget while maximizing recall
```

---

## Dimension 3: Data Pipeline Scaling

### Catalog Ingestion at Scale

```
Product additions: ~100K new products/day (large e-commerce platform)

Naive approach: Embed one at a time → 100K × 20ms = 33 minutes
Batch approach: GPU batch inference, batch_size=512 → ~100K products in ~2 minutes

At Amazon/Alibaba scale: millions of new products per day
  → Need distributed embedding: Spark + GPU clusters
  → Each Spark executor runs CLIP on 1 GPU
  → Process 1M products in ~20 minutes with 10 GPUs
```

### Streaming Pipeline (Near Real-Time)

```
New product added to catalog DB
         │
         ▼ (CDC: Change Data Capture)
    Kafka topic: product.new
         │
         ▼
    Embedding Worker (Flink / Kafka Streams)
    - Consumes from topic
    - Runs CLIP inference (GPU)
    - Publishes embedding to: embedding.product topic
         │
         ▼
    Index Update Service
    - Consumes embeddings
    - Appends to current FAISS index (or updates Milvus)
    - New product searchable within ~5 minutes
```

### Training Data Pipeline at Scale

```
User events: 100M clicks/day, 1M purchases/day

Event ingestion: Kafka → Flink (real-time processing)
  - Join click events with search queries (what was the query image?)
  - Enrich with product metadata
  - Write to training data store (Hive / BigQuery)

Weekly training data prep:
  - Pull last 30 days of click data (3B events)
  - Filter to high-confidence positives (viewed > 2s, clicked, purchased)
  - Mine hard negatives (top-100 ANN results that were NOT clicked)
  - Generate triplets: (query_embed, positive_embed, negative_embed)
  - Write to Parquet shards for training
```

---

## Scaling Cost Optimization

| Optimization | Cost Reduction | Quality Impact |
|---|---|---|
| Spot instances for batch jobs | 70% reduction | None |
| GPU sharing (multiple models per GPU) | 50% reduction | Slight latency increase |
| INT8 quantization (GPU) | 50% reduction | <1% Recall@10 loss |
| Cache common query embeddings | 20-40% reduction | None |
| Smaller embedding dim (512→256) | 30% index cost reduction | ~2% Recall@10 loss |
| Mixed-precision training (FP16) | 50% training cost | None |

---

## Scaling Milestones

| Scale | Catalog Size | QPS | Key Technology |
|---|---|---|---|
| **MVP** | 10K products | 10 QPS | Single GPU, FAISS Flat, no caching |
| **Startup** | 1M products | 100 QPS | 2 GPUs, FAISS IVFFlat, basic caching |
| **Growth** | 10M products | 1K QPS | GPU pod scaling, sharded FAISS, Redis cache |
| **Scale** | 100M products | 10K QPS | Distributed FAISS, distilled models, streaming indexing |
| **Hyper-scale** | 1B+ products | 100K QPS | Custom ANN (ScaNN), full distillation stack, multi-region |

---

## Interview Checkpoint

1. **"How do you scale from 1M to 100M products?"**
   - FAISS index sharding, distributed embedding batch jobs, streaming indexing for freshness, horizontal scaling of GPU serving pods.

2. **"What's the cost structure of this system?"**
   - Dominated by GPU inference (embedding service). Use distillation + INT8 quantization + request batching to maximize GPU utilization. Spot instances for batch jobs.

3. **"How do you handle Black Friday traffic (10x normal QPS)?"**
   - Auto-scaling with pre-warmed GPU pods (pre-scale 2 hours before expected peak). Cache popular queries (same product page images get identical queries). CDN layer for static results.

4. **"If I gave you 10x compute budget, where would you invest it?"**
   - Priority: (1) Larger base model (ViT-L/14 for 4% quality gain), (2) Better hard negative mining (3x more compute for offline mining), (3) Streaming real-time index (reduce freshness lag from 1 hour to 5 minutes).
