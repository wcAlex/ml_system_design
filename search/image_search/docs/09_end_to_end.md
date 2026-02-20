# 09 — End-to-End System Design + Mini Implementation

## Full System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OFFLINE (BATCH) PIPELINE                          │
│                                                                             │
│  Product DB ──► Image Downloader ──► GPU Batch Embedding Job                │
│                                              │                              │
│                                              ▼                              │
│                                    Embedding Store (S3)                     │
│                                              │                              │
│                                              ▼                              │
│                                    ANN Index Builder (FAISS)                │
│                                              │                              │
│                                              ▼                              │
│                                    Product FAISS Index (nightly rebuild)    │
└─────────────────────────────────────────────────────────────────────────────┘
                                               │
                                    (loaded into memory)
                                               │
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ONLINE (REAL-TIME) PIPELINE                       │
│                                                                             │
│  User uploads image                                                         │
│       │                                                                     │
│       ▼                                                                     │
│  API Gateway ──► Rate limiter ──► Image validator                           │
│       │                                                                     │
│       ▼                                                                     │
│  Preprocessing Service                                                      │
│  - Resize to 224×224                                                        │
│  - EXIF rotation fix                                                        │
│  - Object detection (optional: select crop)                                 │
│       │                                                                     │
│       ▼                                                                     │
│  Query Embedding Service (GPU)                    ┌─────────────────┐      │
│  - CLIP ViT-B/16 inference                        │  User Feature   │      │
│  - Returns 512-d L2-normalized vector             │  Service        │      │
│       │                                           │  (user history, │      │
│       │                                           │  preferences)   │      │
│       ▼                                           └────────┬────────┘      │
│  ANN Retrieval Service                                      │               │
│  - FAISS IVF-PQ index                                       │               │
│  - Returns top-500 (product_id, distance) pairs             │               │
│       │                                                     │               │
│       └──────────────────────┬──────────────────────────────┘              │
│                              ▼                                              │
│  Re-ranking Service                                                         │
│  - Apply hard filters (in-stock, price range, category)                    │
│  - Score: visual_sim + personalization + business rules                     │
│  - Return top-20                                                            │
│       │                                                                     │
│       ▼                                                                     │
│  Response + Logging                                                         │
│  - Return JSON to client                                                    │
│  - Log (query_embedding, results, user_id) for training feedback            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Mini Implementation

A self-contained Python demo with CLIP + FAISS. Run this to see the system end-to-end.

### Setup

```bash
cd image_search
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### `src/indexer.py` — Build Product Index

```python
"""
indexer.py — Build FAISS index from product images
"""
import os
import json
import numpy as np
import faiss
import torch
import clip
from PIL import Image
from pathlib import Path
from typing import List, Dict


class ProductIndexer:
    """
    Encodes product images and builds a FAISS index for similarity search.
    """

    def __init__(self, model_name: str = "ViT-B/16", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self.embedding_dim = 512  # ViT-B/16 output dimension

    @torch.no_grad()
    def encode_images(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of images into L2-normalized embeddings.
        Returns: (N, D) float32 array
        """
        all_embeddings = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(self.preprocess(img))
                except Exception as e:
                    print(f"Warning: could not load {path}: {e}")
                    images.append(torch.zeros(3, 224, 224))  # zero placeholder

            image_tensor = torch.stack(images).to(self.device)
            embeddings = self.model.encode_image(image_tensor)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # L2 normalize
            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)

    def build_index(
        self,
        embeddings: np.ndarray,
        index_type: str = "IVFFlat"
    ) -> faiss.Index:
        """
        Build a FAISS index.

        index_type options:
          - "Flat":     exact search (baseline, slow at scale)
          - "IVFFlat":  inverted file, fast approximate search
          - "IVFPQ":    compressed, memory-efficient
          - "HNSW":     graph-based, lowest latency
        """
        d = embeddings.shape[1]
        n = embeddings.shape[0]

        if index_type == "Flat":
            index = faiss.IndexFlatIP(d)  # Inner product = cosine for L2-normalized

        elif index_type == "IVFFlat":
            nlist = min(int(4 * np.sqrt(n)), 1000)
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)

        elif index_type == "IVFPQ":
            nlist = min(int(4 * np.sqrt(n)), 1000)
            m = 32  # number of sub-quantizers
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
            index.train(embeddings)

        elif index_type == "HNSW":
            index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200

        else:
            raise ValueError(f"Unknown index type: {index_type}")

        index.add(embeddings)
        print(f"Index built: {index.ntotal} vectors, type={index_type}")
        return index

    def save_index(self, index: faiss.Index, path: str):
        faiss.write_index(index, path)
        print(f"Index saved to {path}")

    def load_index(self, path: str) -> faiss.Index:
        return faiss.read_index(path)


def build_catalog_index(
    catalog_dir: str,
    output_dir: str,
    index_type: str = "IVFFlat"
):
    """
    Build and save index from a directory of product images.

    catalog_dir structure:
        catalog/
            product_001.jpg  (filename = product_id)
            product_002.jpg
            ...
    """
    os.makedirs(output_dir, exist_ok=True)

    image_paths = sorted(Path(catalog_dir).glob("*.jpg")) + \
                  sorted(Path(catalog_dir).glob("*.png"))
    product_ids = [p.stem for p in image_paths]  # filename without extension

    print(f"Found {len(image_paths)} products")

    indexer = ProductIndexer()
    embeddings = indexer.encode_images([str(p) for p in image_paths])

    index = indexer.build_index(embeddings, index_type=index_type)
    indexer.save_index(index, os.path.join(output_dir, "product.index"))

    # Save product ID mapping (index position → product_id)
    id_map = {i: pid for i, pid in enumerate(product_ids)}
    with open(os.path.join(output_dir, "id_map.json"), "w") as f:
        json.dump(id_map, f)

    # Save embeddings for re-use (e.g., re-ranking)
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)

    print(f"Done. Index, id_map, embeddings saved to {output_dir}")
    return index, id_map, embeddings
```

### `src/searcher.py` — Query-Time Search

```python
"""
searcher.py — Query image → ranked product results
"""
import json
import numpy as np
import faiss
import torch
import clip
from PIL import Image
from typing import List, Tuple, Optional, Dict


class VisualSearcher:
    """
    Encodes a query image and retrieves similar products from the FAISS index.
    """

    def __init__(
        self,
        index_path: str,
        id_map_path: str,
        model_name: str = "ViT-B/16",
        nprobe: int = 50,
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        self.index = faiss.read_index(index_path)
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe  # higher = more accurate, slower

        with open(id_map_path) as f:
            raw = json.load(f)
            self.id_map = {int(k): v for k, v in raw.items()}

    @torch.no_grad()
    def encode_query(self, image: Image.Image) -> np.ndarray:
        """Encode a PIL image → L2-normalized 512-d embedding."""
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        embedding = self.model.encode_image(tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().astype(np.float32)

    def search(
        self,
        query_image: Image.Image,
        top_k: int = 20,
        retrieve_k: int = 500
    ) -> List[Dict]:
        """
        Search for visually similar products.

        Returns: list of {product_id, score, rank}
        """
        query_embedding = self.encode_query(query_image)
        distances, indices = self.index.search(query_embedding, retrieve_k)

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for padded results
                continue
            product_id = self.id_map.get(idx, f"unknown_{idx}")
            results.append({
                "rank": rank + 1,
                "product_id": product_id,
                "score": float(dist),  # cosine similarity (0 to 1 for L2-normalized)
                "index_position": int(idx)
            })

        return results[:top_k]

    def search_with_text(
        self,
        query_image: Image.Image,
        text_modifier: str,
        top_k: int = 20,
        retrieve_k: int = 500,
        image_weight: float = 0.7
    ) -> List[Dict]:
        """
        Multi-modal search: combine image embedding with text modifier.
        Example: image of a shoe + text "in red color"
        """
        # Image embedding
        img_embedding = self.encode_query(query_image)

        # Text embedding
        with torch.no_grad():
            text_tokens = clip.tokenize([text_modifier]).to(self.device)
            text_embedding = self.model.encode_text(text_tokens)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            text_embedding = text_embedding.cpu().numpy().astype(np.float32)

        # Weighted fusion
        fused = image_weight * img_embedding + (1 - image_weight) * text_embedding
        fused = fused / np.linalg.norm(fused, axis=-1, keepdims=True)

        distances, indices = self.index.search(fused, retrieve_k)

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:
                continue
            results.append({
                "rank": rank + 1,
                "product_id": self.id_map.get(idx, f"unknown_{idx}"),
                "score": float(dist)
            })

        return results[:top_k]
```

### `src/reranker.py` — Re-ranking Layer

```python
"""
reranker.py — Re-rank ANN candidates with business rules and personalization
"""
from typing import List, Dict, Optional
import numpy as np


class ReRanker:
    """
    Applies filtering and scoring on ANN candidates to produce final results.

    Scoring formula:
        final_score = α * visual_score
                    + β * attribute_match
                    + γ * personalization_score
                    + δ * freshness_score
                    - ε * price_distance_penalty
    """

    def __init__(
        self,
        alpha: float = 0.6,   # visual similarity weight
        beta: float = 0.15,   # attribute match weight
        gamma: float = 0.15,  # personalization weight
        delta: float = 0.05,  # freshness weight
        epsilon: float = 0.05 # price penalty weight
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon

    def rerank(
        self,
        candidates: List[Dict],
        product_metadata: Dict[str, Dict],
        user_context: Optional[Dict] = None,
        filters: Optional[Dict] = None,
        top_k: int = 20
    ) -> List[Dict]:
        """
        Re-rank candidates.

        candidates: list of {product_id, score (visual similarity), rank}
        product_metadata: {product_id: {price, category, in_stock, days_since_added, ...}}
        user_context: {preferred_price_range, preferred_categories, purchase_history}
        filters: {max_price, min_price, category, in_stock_only}
        """
        results = []

        for candidate in candidates:
            pid = candidate["product_id"]
            meta = product_metadata.get(pid, {})

            # Hard filters — remove ineligible products
            if filters:
                if filters.get("in_stock_only") and not meta.get("in_stock", True):
                    continue
                price = meta.get("price", 0)
                if filters.get("max_price") and price > filters["max_price"]:
                    continue
                if filters.get("min_price") and price < filters["min_price"]:
                    continue
                if filters.get("category") and meta.get("category") != filters["category"]:
                    continue

            # Visual similarity score (from ANN)
            visual_score = candidate["score"]

            # Attribute match score
            attr_score = self._attribute_match(candidate, meta, user_context)

            # Personalization score
            pers_score = self._personalization_score(pid, user_context)

            # Freshness bonus (newer products get a small boost)
            days_old = meta.get("days_since_added", 365)
            freshness = np.exp(-days_old / 30)  # decay over ~30 days

            # Price distance penalty
            price_penalty = self._price_distance_penalty(meta, user_context)

            # Final score
            final_score = (
                self.alpha * visual_score
                + self.beta * attr_score
                + self.gamma * pers_score
                + self.delta * freshness
                - self.epsilon * price_penalty
            )

            results.append({
                **candidate,
                "final_score": final_score,
                "visual_score": visual_score,
                "attr_score": attr_score,
                "pers_score": pers_score,
                "freshness_score": freshness,
                "metadata": meta
            })

        # Sort by final score descending
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]

    def _attribute_match(self, candidate, meta, user_context):
        if not user_context or not meta:
            return 0.5  # neutral
        match_score = 0.0
        if user_context.get("preferred_category") == meta.get("category"):
            match_score += 0.5
        if user_context.get("preferred_color") and \
           user_context["preferred_color"] in meta.get("colors", []):
            match_score += 0.5
        return match_score

    def _personalization_score(self, product_id, user_context):
        if not user_context:
            return 0.0
        # Boost items from categories the user has bought from
        preferred = user_context.get("preferred_category_ids", set())
        if product_id in preferred:
            return 1.0
        return 0.0

    def _price_distance_penalty(self, meta, user_context):
        if not user_context or "typical_spend" not in user_context:
            return 0.0
        price = meta.get("price", 0)
        typical = user_context["typical_spend"]
        relative_diff = abs(price - typical) / (typical + 1e-6)
        return min(relative_diff, 1.0)  # cap at 1
```

### `src/demo.py` — End-to-End Demo

```python
"""
demo.py — Run the full visual search pipeline end-to-end
"""
import os
import sys
from PIL import Image
from pathlib import Path

# Ensure src/ is in path
sys.path.insert(0, os.path.dirname(__file__))

from indexer import build_catalog_index, ProductIndexer
from searcher import VisualSearcher
from reranker import ReRanker


def run_demo():
    """
    Demo workflow:
    1. Build index from sample catalog images
    2. Run a visual query
    3. Re-rank results

    For a real demo, provide actual product images.
    """
    catalog_dir = "data/catalog"
    index_dir = "data/index"

    # Step 1: Build index (run once)
    if not Path(f"{index_dir}/product.index").exists():
        print("Building catalog index...")
        build_catalog_index(catalog_dir, index_dir, index_type="IVFFlat")
    else:
        print("Index already exists. Loading...")

    # Step 2: Initialize searcher
    searcher = VisualSearcher(
        index_path=f"{index_dir}/product.index",
        id_map_path=f"{index_dir}/id_map.json",
        nprobe=50
    )

    # Step 3: Run a visual query
    query_image_path = "data/query.jpg"  # Replace with actual query image
    if not Path(query_image_path).exists():
        print(f"Query image not found at {query_image_path}")
        print("Please add a query image to run the demo.")
        return

    query_image = Image.open(query_image_path).convert("RGB")
    print(f"\nSearching with query image: {query_image_path}")

    # Image-only search
    results = searcher.search(query_image, top_k=20, retrieve_k=100)
    print(f"\nTop-5 visual search results (before re-ranking):")
    for r in results[:5]:
        print(f"  Rank {r['rank']}: {r['product_id']} (score={r['score']:.4f})")

    # Multi-modal search (image + text)
    print("\nMulti-modal search: query image + 'in red color'")
    results_mm = searcher.search_with_text(
        query_image,
        text_modifier="in red color",
        top_k=20,
        image_weight=0.7
    )
    print(f"Top-5 multi-modal results:")
    for r in results_mm[:5]:
        print(f"  Rank {r['rank']}: {r['product_id']} (score={r['score']:.4f})")

    # Step 4: Re-rank
    # Simulated product metadata and user context
    product_metadata = {r["product_id"]: {
        "price": 100.0,
        "category": "shoes",
        "in_stock": True,
        "days_since_added": 10,
        "colors": ["white", "black"]
    } for r in results}

    user_context = {
        "preferred_category": "shoes",
        "typical_spend": 120.0
    }

    reranker = ReRanker()
    final_results = reranker.rerank(
        candidates=results,
        product_metadata=product_metadata,
        user_context=user_context,
        filters={"in_stock_only": True},
        top_k=10
    )

    print(f"\nTop-5 re-ranked results:")
    for r in final_results[:5]:
        print(f"  {r['product_id']}: final={r['final_score']:.4f}, "
              f"visual={r['visual_score']:.4f}, pers={r['pers_score']:.4f}")


if __name__ == "__main__":
    run_demo()
```

---

## Data Flow Summary

```
1. [Offline] Product images → CLIP encoding → 512-d embeddings
2. [Offline] Embeddings → FAISS IVFFlat index (nightly rebuild)
3. [Online]  User uploads photo → resize + preprocess
4. [Online]  Query → CLIP encoding → 512-d query vector
5. [Online]  Query vector → FAISS search (nprobe=50) → top-500 candidates
6. [Online]  500 candidates → hard filters → attribute scoring → personalization
7. [Online]  Top-20 results → returned to user (< 200ms total)
8. [Async]   User clicks logged → used for future training data
```

---

## Interview Checkpoint

1. **"Walk me through what happens when a user uploads an image."**
   - The 8-step data flow above. Practice narrating this end-to-end in under 2 minutes.

2. **"What's in your serving stack?"**
   - API Gateway → Preprocessing → CLIP GPU inference → FAISS retrieval → Re-ranking service → Response

3. **"How do you test this system before deploying?"**
   - Unit test each service. Integration test with sample queries against offline index. Shadow mode (run new model in parallel with old). Then A/B test.

4. **"If you had to cut latency in half, what would you do?"**
   - Distill CLIP to a smaller model. Use quantization (INT8). Switch from IVFFlat to HNSW index. Cache common queries. Profile to find the actual bottleneck first.
