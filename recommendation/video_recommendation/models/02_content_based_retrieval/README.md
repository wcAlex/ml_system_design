# Content-Based Retrieval Model

## 1. Role in the Pipeline

Content-Based Retrieval is the **second retrieval source** in Stage 1, generating ~100 candidates based on video content similarity to the user's recent watch history. It complements the Two-Tower model by:

- **Handling item cold-start**: New videos with zero interactions can still be retrieved if their content is similar to what the user watches
- **Capturing content preferences**: A user who watches Python tutorials should see similar Python content even if CF hasn't learned this yet
- **Providing diversity**: Surfaces content similar in topic/style but from different creators or sub-communities

```
User's recent watch history          Video corpus
  [v1, v2, v3, ..., v20]            [all videos]
         ↓                                ↓
  Aggregate content embeddings       Precomputed content embeddings
         ↓                                ↓
    user_content_profile             item_content_emb
         ↓                                ↓
         └────── ANN search ──────────────┘
                     ↓
              Top-100 similar videos
```

---

## 2. Data Requirements

### Content Signals Per Video

| Signal | Source | Processing |
|--------|--------|------------|
| Title | Video metadata | Text embedding |
| Description | Video metadata | Text embedding |
| Tags / Labels | Video metadata or ML classifier | Multi-hot or embedding |
| Transcript (ASR) | Speech-to-text | Text embedding |
| Thumbnail | Video metadata | Visual embedding (CNN) |
| Audio features | Audio track | Audio embedding |
| Video frames | Sampled keyframes | Visual embedding (ViT / CLIP) |

### User Watch History

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | string | User identifier |
| `video_id` | string | Video watched |
| `watch_time_sec` | float | Duration watched |
| `timestamp` | int64 | When watched |

### Example Raw Data

**Video content record:**
```json
{
  "video_id": "v_19283",
  "title": "How to Train a Neural Network from Scratch",
  "description": "Step by step guide to building and training...",
  "tags": ["machine learning", "neural networks", "python", "tutorial"],
  "duration_sec": 1245,
  "category": "education",
  "language": "en",
  "transcript_snippet": "Welcome to this tutorial on neural networks...",
  "thumbnail_url": "https://..."
}
```

**User watch history:**
```json
{"user_id": "u_382910", "video_id": "v_19283", "watch_time_sec": 1100, "timestamp": 1700000000}
{"user_id": "u_382910", "video_id": "v_44521", "watch_time_sec": 890, "timestamp": 1700003600}
```

---

## 3. Feature Engineering

### Video Content Embedding Pipeline

#### Option A: Text-Only Embedding (Simple, Effective)

```
title + description + tags + transcript
          ↓
  Pretrained Sentence Transformer
  (e.g., all-MiniLM-L6-v2 or E5-large)
          ↓
  text_emb (384-d or 768-d)
          ↓
  Optional: PCA to 128-d
          ↓
  video_content_emb (128-d)
```

**Pros**: Simple, fast, good for text-rich content
**Cons**: Misses visual/audio signals

#### Option B: Multimodal Embedding (CLIP-based)

```
  title + description              thumbnail / keyframes
        ↓                                  ↓
  CLIP text encoder               CLIP image encoder
        ↓                                  ↓
  text_emb (512-d)               visual_emb (512-d)
        ↓                                  ↓
        └───── concatenate + project ──────┘
                       ↓
                  MLP(1024 → 256)
                       ↓
               video_content_emb (256-d)
```

**Pros**: Captures both textual and visual content, handles videos with sparse text
**Cons**: Higher compute cost, CLIP model is large

#### Option C: Fine-Tuned Domain Embedding (Best Quality)

Start from pretrained model (CLIP or sentence-transformer), fine-tune on platform data:
- Positive pairs: videos watched in same session by same user
- Negative pairs: random videos
- Contrastive loss (similar to Two-Tower training)

**Pros**: Embeddings aligned with actual user preferences
**Cons**: Requires training infrastructure, periodic retraining

### Comparison

| Approach | Quality | Compute Cost | Cold-Start | Maintenance |
|----------|---------|-------------|------------|-------------|
| Text-only | Good | Low | Excellent | Low |
| CLIP multimodal | Better | Medium | Excellent | Medium |
| Fine-tuned | Best | High | Excellent | High |

**Recommendation**: Start with Option A, graduate to Option C as the system matures.

### User Profile Construction

Aggregate recent watch history into a user content profile:

```python
def build_user_content_profile(user_history, video_embeddings,
                                max_history=50, decay=0.95):
    """
    Build user content profile from watch history with time decay.

    More recent videos get higher weight via exponential decay.
    Watch time also acts as an engagement weight.
    """
    if not user_history:
        return None  # Cold-start user → fall back to other sources

    # Sort by recency (most recent first)
    history = sorted(user_history, key=lambda x: x["timestamp"], reverse=True)
    history = history[:max_history]

    weighted_embs = []
    weights = []

    for i, event in enumerate(history):
        vid = event["video_id"]
        if vid not in video_embeddings:
            continue

        time_weight = decay ** i  # Exponential decay by recency
        engagement_weight = min(event["watch_time_sec"] / 60.0, 10.0)  # Cap at 10
        w = time_weight * engagement_weight

        weighted_embs.append(video_embeddings[vid] * w)
        weights.append(w)

    if not weighted_embs:
        return None

    # Weighted average
    profile = sum(weighted_embs) / sum(weights)
    # L2 normalize for cosine similarity search
    profile = profile / np.linalg.norm(profile)
    return profile
```

**Key design decisions:**
- **Time decay**: Recent watches matter more than old ones
- **Watch time weighting**: Longer engagement = stronger signal
- **Cap history at 50**: Keeps profile fresh, prevents stale interests from dominating

---

## 4. Model Architecture

Content-based retrieval is **not a traditional supervised model** — it's an embedding + similarity pipeline. The "model" components are:

### Component 1: Content Encoder

Converts raw video content into a dense embedding.

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class VideoContentEncoder(nn.Module):
    """Encode video text content into a dense embedding."""

    def __init__(self, pretrained_model="sentence-transformers/all-MiniLM-L6-v2",
                 output_dim=128):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        hidden_dim = self.encoder.config.hidden_size  # 384 for MiniLM

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=256, return_tensors="pt"
        )
        with torch.no_grad():
            output = self.encoder(**tokens)
        # Mean pooling
        emb = output.last_hidden_state.mean(dim=1)
        return emb

    def forward(self, texts: list[str]) -> torch.Tensor:
        raw_emb = self.encode_text(texts)
        projected = self.projection(raw_emb)
        return nn.functional.normalize(projected, p=2, dim=1)


def precompute_video_embeddings(encoder, video_metadata_list):
    """Batch-compute embeddings for all videos in the corpus."""
    embeddings = {}

    for batch in chunk(video_metadata_list, batch_size=256):
        texts = [
            f"{v['title']}. {v['description']}. Tags: {', '.join(v['tags'])}"
            for v in batch
        ]
        embs = encoder(texts)

        for v, emb in zip(batch, embs):
            embeddings[v["video_id"]] = emb.cpu().numpy()

    return embeddings
```

### Component 2: Optional Fine-Tuning with Contrastive Loss

If you want to fine-tune the encoder on platform-specific data:

```python
class ContrastiveContentModel(nn.Module):
    """Fine-tune content encoder with session co-watch pairs."""

    def __init__(self, encoder: VideoContentEncoder, temperature=0.07):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature

    def forward(self, anchor_texts, positive_texts):
        """
        anchor_texts: list of texts for anchor videos
        positive_texts: list of texts for co-watched videos (same session)
        """
        anchor_emb = self.encoder(anchor_texts)    # (B, D)
        pos_emb = self.encoder(positive_texts)      # (B, D)

        # In-batch contrastive (InfoNCE)
        logits = torch.matmul(anchor_emb, pos_emb.T) / self.temperature
        labels = torch.arange(len(anchor_texts), device=logits.device)
        loss = nn.functional.cross_entropy(logits, labels)

        return loss


def train_content_encoder(model, session_pairs_loader, optimizer, epochs=3):
    """
    session_pairs_loader yields batches of (anchor_video_text, positive_video_text)
    where positive = another video watched in same session by same user.
    """
    for epoch in range(epochs):
        total_loss = 0
        for batch in session_pairs_loader:
            loss = model(batch["anchor_text"], batch["positive_text"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: loss={total_loss/len(session_pairs_loader):.4f}")
```

---

## 5. Retrieval Pipeline (End-to-End)

```python
import faiss
import numpy as np


class ContentBasedRetriever:
    """Full content-based retrieval pipeline."""

    def __init__(self, video_embeddings: dict, embedding_dim: int = 128):
        self.video_ids = list(video_embeddings.keys())
        self.embeddings = np.stack(
            [video_embeddings[vid] for vid in self.video_ids]
        ).astype("float32")

        # Build FAISS index
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.index.add(self.embeddings)

    def retrieve(self, user_content_profile: np.ndarray,
                 top_k: int = 100,
                 exclude_video_ids: set = None) -> list[dict]:
        """
        Retrieve top-K videos similar to user's content profile.

        Args:
            user_content_profile: (1, D) normalized embedding
            top_k: number of candidates to return
            exclude_video_ids: videos to filter out (already watched)
        """
        # Over-fetch to account for filtering
        fetch_k = top_k * 2 if exclude_video_ids else top_k

        query = user_content_profile.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(query, fetch_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            vid = self.video_ids[idx]
            if exclude_video_ids and vid in exclude_video_ids:
                continue
            results.append({"video_id": vid, "score": float(score)})
            if len(results) >= top_k:
                break

        return results
```

---

## 6. Model Input / Output Examples

### Embedding Input

**Video metadata input:**
```python
video = {
    "video_id": "v_19283",
    "title": "How to Train a Neural Network from Scratch",
    "description": "Step by step guide to building neural networks using PyTorch",
    "tags": ["machine learning", "neural networks", "python", "tutorial"],
}

# Concatenated text fed to encoder:
text = "How to Train a Neural Network from Scratch. Step by step guide to building neural networks using PyTorch. Tags: machine learning, neural networks, python, tutorial"
```

**Embedding output:**
```python
video_emb = np.array([0.032, -0.118, 0.045, ..., 0.089])  # shape: (128,)
```

### Retrieval Input/Output

**Input**: User content profile (aggregated from watch history)
```python
user_profile = np.array([0.041, -0.097, 0.062, ..., 0.075])  # shape: (128,)
# Built from user's recent watches: ML tutorials, Python guides, data science talks
```

**Output**: Top-100 content-similar candidates
```python
candidates = [
    {"video_id": "v_30291", "score": 0.94},   # "PyTorch Lightning Tutorial"
    {"video_id": "v_88712", "score": 0.91},   # "Deep Learning Fundamentals"
    {"video_id": "v_12093", "score": 0.89},   # "Neural Network Math Explained"
    {"video_id": "v_55432", "score": 0.87},   # "TensorFlow vs PyTorch 2024"
    # ... 96 more
]
```

---

## 7. Evaluation Methods

### Offline Metrics

| Metric | Description | How to Compute |
|--------|-------------|----------------|
| **Recall@K** (K=100) | Of videos user will watch next, how many are in top-K? | Hold out future watches, check overlap |
| **Content Diversity** | How diverse are the retrieved candidates? | Average pairwise cosine distance in result set |
| **Cold-Start Recall** | Recall@K restricted to videos with < 10 interactions | Critical metric for content-based advantage |
| **Embedding Quality (Intrinsic)** | Do similar videos have similar embeddings? | Manual annotation of video pairs + rank correlation |

### Key Evaluation: Cold-Start Comparison

The content-based model should **significantly outperform** collaborative filtering on cold-start items:

```python
def evaluate_cold_start_recall(retriever, test_users, ground_truth,
                                cold_videos, k=100):
    """
    Evaluate recall only on cold-start videos (< N interactions).
    This is where content-based shines vs. collaborative filtering.
    """
    recalls = []
    for user_id, profile in test_users.items():
        candidates = retriever.retrieve(profile, top_k=k)
        retrieved = set(c["video_id"] for c in candidates)

        # Only measure against cold-start ground truth
        relevant_cold = set(ground_truth[user_id]) & cold_videos
        if relevant_cold:
            recall = len(retrieved & relevant_cold) / len(relevant_cold)
            recalls.append(recall)

    return np.mean(recalls)
```

### Embedding Quality Evaluation

```python
def evaluate_embedding_quality(embeddings, category_labels):
    """
    Check if videos in the same category cluster together.
    Uses silhouette score as a proxy for embedding quality.
    """
    from sklearn.metrics import silhouette_score

    vids = list(embeddings.keys())
    X = np.stack([embeddings[v] for v in vids])
    labels = [category_labels[v] for v in vids]

    score = silhouette_score(X, labels, metric="cosine", sample_size=10000)
    print(f"Silhouette score (by category): {score:.3f}")
    # > 0.3 is decent, > 0.5 is good
    return score
```

### Online Evaluation (A/B Test)

| Metric | What to Measure |
|--------|----------------|
| Coverage of new videos | % of videos < 24h old that get at least 1 impression |
| Long-tail engagement | Watch time on videos outside top-1000 most popular |
| Content diversity | Intra-list diversity of final recommendations |

---

## 8. Interview Talking Points

1. **Why content-based alongside collaborative filtering?**
   - CF requires interactions → fails on new videos (cold-start)
   - Content-based uses metadata → works from day zero
   - Together they cover both personalization depth (CF) and breadth/freshness (content)

2. **Embedding model choice**
   - Start with pretrained sentence-transformers (zero training needed)
   - Fine-tune with contrastive learning on co-watch pairs for better domain alignment
   - Consider CLIP for multimodal (thumbnail + text) when text is sparse

3. **User profile aggregation matters**
   - Simple average of history embeddings is a strong baseline
   - Time-weighted average (decay) captures interest drift
   - Multi-interest: cluster history into K centroids → query each (used by TDM/Alibaba)

4. **Limitations and mitigations**
   - Content-based creates "filter bubbles" → mitigate with diversity in re-ranking
   - Text embeddings may miss production quality (good content ≠ good video)
   - Combine with engagement signals in the ranking stage

5. **Scaling: embedding computation cost**
   - Precompute all video embeddings offline (batch job)
   - Only compute embeddings for new videos in near-real-time
   - Use distilled/small models (MiniLM-L6 = 22M params vs BERT-base = 110M params)
