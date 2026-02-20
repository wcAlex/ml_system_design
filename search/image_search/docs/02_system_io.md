# 02 — System Input & Output

## Define the Contract First

In an interview, sketch the I/O before the architecture. It forces clarity about what data you have, what latency you must meet, and where complexity lives.

---

## Inputs

### Query-Time Inputs (per search request)

| Input | Type | Notes |
|---|---|---|
| **Query image** | JPEG/PNG, up to 5MB | Primary signal — could be a photo, screenshot, or cropped region |
| **Optional text refinement** | String, e.g. "but in red" | Multi-modal query (CLIP enables this) |
| **User ID** | String (optional, anonymous OK) | Used for personalization |
| **Session context** | Recent views, clicks in session | Short-term preference signal |
| **Filters** | Category, price range, brand | Applied post-retrieval or as hard constraints |
| **Device/locale** | Mobile vs desktop, country | Affects ranking (shipping eligibility, language) |

### Catalog-Time Inputs (offline indexing pipeline)

| Input | Type | Notes |
|---|---|---|
| **Product images** | Multiple angles per SKU | Usually 5–10 images per product |
| **Product metadata** | Title, description, category, price, brand | Used for filtering and re-ranking |
| **Inventory status** | In-stock bool | Filter out unavailable items |
| **Seller quality score** | Float | Business rule for ranking |

---

## Outputs

### Primary Output (ranked result list)

```json
{
  "query_id": "abc123",
  "results": [
    {
      "product_id": "SKU-789",
      "score": 0.94,
      "title": "Air Force 1 Low '07",
      "image_url": "https://...",
      "price": 110.00,
      "brand": "Nike",
      "in_stock": true,
      "visual_similarity_score": 0.94,
      "personalization_boost": 0.03
    }
  ],
  "latency_ms": 145,
  "retrieval_count": 500,
  "returned_count": 20
}
```

### Key design decisions in the output:
- **Score decomposition:** Separate visual_similarity vs personalization_boost helps with debugging and A/B testing
- **Retrieval count vs returned count:** You retrieve ~500 candidates (ANN), re-rank, return top 20 to UI
- **latency_ms:** Log this — P50/P95/P99 monitoring is critical

---

## System Boundary Diagram

```
User (mobile app / browser)
         │
         │  POST /search/visual
         │  {image: <bytes>, user_id, filters}
         ▼
┌─────────────────────────────────┐
│         API Gateway             │  - Auth, rate limiting, image validation
│         (Load Balancer)         │  - Resize image to 224x224 (or 336x336)
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│      Query Embedding Service    │  - Run image through encoder model
│      (GPU inference, ~20ms)     │  - Returns 512-d or 768-d vector
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│      ANN Retrieval Service      │  - FAISS / ScaNN index
│      (~10M products, ~30ms)     │  - Returns top-500 candidates + distances
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│      Re-ranking Service         │  - Apply filters (stock, price, category)
│      (~50ms)                    │  - Personalization boost
│                                 │  - Business rules (sponsored, quality score)
└────────────┬────────────────────┘
             │
             ▼
         Top-K results
         returned to user
```

---

## Latency Budget (P99 = 200ms)

| Stage | Budget | Notes |
|---|---|---|
| Network + API gateway | ~20ms | TLS handshake, routing |
| Image pre-processing | ~5ms | Resize, normalize, CPU |
| Query embedding (GPU) | ~20–40ms | Model inference |
| ANN search | ~20–40ms | Depends on index size and sharding |
| Re-ranking | ~30–50ms | Small neural model or rule-based |
| Response serialization | ~5ms | |
| **Total** | **~100–160ms** | Leaves headroom for P99 spikes |

**Key insight:** The embedding step dominates if you use large models. A ViT-L is 2-4x slower than a ViT-B. This is a core trade-off in model choice (doc 03).

---

## API Contract

### Endpoint
```
POST /v1/search/visual
Content-Type: multipart/form-data

Fields:
  image        (required) binary image data
  user_id      (optional) string
  category     (optional) filter: "shoes", "tops", etc.
  max_price    (optional) float
  min_price    (optional) float
  limit        (optional, default=20) int, max=100
```

### Error Responses
| Code | Meaning |
|---|---|
| 400 | Invalid image (wrong format, too large, no detectable object) |
| 429 | Rate limit exceeded |
| 503 | Index temporarily unavailable (fallback: return popular items) |

---

## Offline Pipeline Boundary

```
Product Catalog DB ──► Image Downloader ──► Embedding Batch Job ──► ANN Index Builder
                                                                          │
                                                                          ▼
                                                                     FAISS Index
                                                                  (updated hourly)
```

New products flow through this pipeline and appear in search results within ~1 hour (freshness SLA).

---

## Interview Checkpoint

1. **"What happens when the user uploads a photo with multiple objects?"**
   - Options: use the full image (simplest), detect objects with YOLO/Detectron2 and let user select, or return results for the most prominent detected object. Pinterest does region-of-interest selection.

2. **"How do you handle image quality issues (blurry, dark, sideways)?"**
   - Pre-screening: blur detector, NSFW classifier, orientation correction (EXIF + model). Return a 400 with user-friendly message.

3. **"What is your freshness SLA and how do you achieve it?"**
   - Hourly batch vs. streaming (Kafka) for new product ingestion. Trade-off: streaming is fresher but more complex to operate.

4. **"What's your fallback when the visual search service is down?"**
   - Degrade to keyword search using product metadata. Never return an empty page.
