# 03 — Model Choices

## The Embedding Model: Your Most Important Decision

The core of an image search system is the **embedding model** — it converts images into dense vectors where semantically similar images are geometrically close. Everything else (indexing, retrieval, re-ranking) depends on the quality of these embeddings.

---

## Option 1: CNN-Based Embeddings (ResNet / EfficientNet)

### Architecture
```
Input image (224×224×3)
     │
     ▼
Convolutional blocks (feature extraction)
  - Conv + BN + ReLU layers
  - Gradually increase channels (64→2048 in ResNet-50)
  - Spatial downsampling via stride/pooling
     │
     ▼
Global Average Pooling (GAP)
     │
     ▼
Dense embedding layer (e.g., 512-d)
     │
     ▼
L2-normalized vector → used for cosine similarity
```

**Key idea:** Convolutions capture local textures, patterns, and shapes. GAP aggregates spatial information into a fixed-size vector.

### Variants
| Model | Params | ImageNet Top-1 | Latency (GPU) | Notes |
|---|---|---|---|---|
| ResNet-50 | 25M | 76% | ~5ms | Baseline; widely understood |
| EfficientNet-B4 | 19M | 83% | ~8ms | Better accuracy/efficiency ratio |
| EfficientNet-B7 | 66M | 84% | ~20ms | Best CNN accuracy; heavier |

### Pros
- Well-understood, easy to debug
- Strong tooling ecosystem (PyTorch, TF)
- Efficient — runs on CPU for small catalogs
- Works with standard metric learning pipelines

### Cons
- Weak at long-range dependencies (global shape, outfit coherence)
- Requires fine-tuning for domain shift (web-pretrained → product images)
- Fixed receptive field limits understanding of object context

**When to use:** Small-to-medium catalogs (<1M products), latency-sensitive with limited GPU budget, need interpretability.

---

## Option 2: Vision Transformer (ViT)

### Architecture
```
Input image (224×224×3)
     │
     ▼ Patch Embedding
Split into 16×16 patches → 196 patches
Each patch → linear projection → 768-d token
     │
     ▼ Add [CLS] token + positional encoding
     │
     ▼ Transformer Encoder (12–24 layers)
  Multi-head self-attention → global context across all patches
  FFN (feed-forward network)
     │
     ▼ [CLS] token output
     │
     ▼ Projection head → 512-d embedding
```

**Key idea:** Self-attention lets every patch attend to every other patch — captures global structure (whole outfit, object shape) that CNNs miss.

### Variants
| Model | Params | ImageNet Top-1 | GPU Latency | Notes |
|---|---|---|---|---|
| ViT-S/16 | 22M | 81% | ~8ms | Small; matches ResNet-50 accuracy |
| ViT-B/16 | 86M | 85% | ~15ms | Strong baseline |
| ViT-L/16 | 307M | 87% | ~45ms | Best standalone ViT; latency risk |

### Pros
- Captures global image structure (full outfit, scene context)
- Scales better with data than CNNs
- Strong fine-tuning performance with contrastive pretraining

### Cons
- Requires large datasets to train from scratch (CNNs generalize better with small data)
- Higher latency than EfficientNet at same accuracy
- Patch-based: can miss fine-grained textures in small regions

**When to use:** Large product catalogs with diverse categories, you have enough fine-tuning data, latency budget allows ~15–45ms for embedding.

---

## Option 3: CLIP (Contrastive Language-Image Pretraining)

### Architecture
```
Image Encoder          Text Encoder
(ViT or ResNet)        (Transformer)
      │                      │
      ▼                      ▼
 Image embedding         Text embedding
      │                      │
      └──────── Contrastive Loss ────────┘
                (image & caption should be close)
```

**Pretraining:** 400M+ (image, alt-text) pairs from the web. The model learns that an image of "red Nike Air Force 1" should be close to the text "red Nike Air Force 1" in embedding space.

**Why this matters for product search:**
- Product catalogs have natural (image, title) pairs — you can fine-tune CLIP with your catalog
- Enables **multi-modal search**: users can type "like this but in blue" after uploading a photo
- Strong zero-shot: works reasonably for new product categories without retraining

### Variants
| Model | Image Encoder | Params | Notes |
|---|---|---|---|
| CLIP ViT-B/32 | ViT-B | 151M total | Fastest; good baseline |
| CLIP ViT-B/16 | ViT-B | 150M total | Better resolution |
| CLIP ViT-L/14 | ViT-L | 428M total | Best quality; latency cost |
| OpenCLIP (open source) | Various | Various | Reproduces CLIP; more variants |

### Pros
- Multi-modal: handles text + image queries naturally
- Strong zero-shot across categories
- Natural fine-tuning target for product domain
- Used by Google Lens, Amazon, Pinterest at scale

### Cons
- Larger model → higher latency and serving cost
- Text encoder is "wasted" at retrieval time (image-only queries)
- Web-pretrained — knows "Nike shoe" but may not know niche product attributes

**When to use:** This is the current industry standard. If you have multi-modal queries or a broad product catalog, CLIP fine-tuning is the right choice.

---

## Option 4: DINO / DINOv2 (Self-Supervised ViT)

### Architecture
```
Student ViT ──────────────────────────────────────────────────────►  Student output
     ▲                                                                    │
     │ same image, different augmentation                                 │
Teacher ViT (EMA of student weights) ──────────────────────────────► Teacher output
                                                                          │
                                              Self-distillation loss ◄───┘
                                        (student matches teacher's distribution)
```

**Key idea:** No labels required. The teacher (slowly updated EMA copy) generates pseudo-targets for the student. Forces the model to learn meaningful visual representations without human annotation.

**DINOv2 (Meta, 2023):** Trained on curated dataset (LVD-142M). Produces state-of-the-art dense features — better than supervised ImageNet models on many downstream tasks.

### Pros
- No need for labeled product data — critical for new catalogs
- Dense features are excellent for fine-grained retrieval
- DINOv2 beats CLIP on pure image similarity tasks
- Strong for instance-level retrieval (find the exact same item)

### Cons
- Not multi-modal (no text tower)
- Less common in production (newer, less tooling)
- Self-supervised training requires large compute

**When to use:** When you have image-only queries and want the best possible visual similarity without labeled data. Or as a feature extractor on top of CLIP.

---

## Comparison Summary

| Dimension | ResNet/EfficientNet | ViT | CLIP | DINOv2 |
|---|---|---|---|---|
| **Visual similarity quality** | Good | Better | Better | Best (image-only) |
| **Multi-modal (text+image)** | No | No | Yes | No |
| **Zero-shot generalization** | Low | Medium | High | Medium |
| **Training data needed** | Labeled pairs | Labeled pairs | (image, text) pairs | Unlabeled images |
| **Serving latency** | Low | Medium | Medium | Medium |
| **Industry adoption** | Legacy | Growing | Dominant | Emerging |
| **Fine-tuning complexity** | Low | Medium | Medium | Medium |

---

## Industry Standard (2024–2025)

| Company | Approach |
|---|---|
| **Pinterest** | Custom ViT + contrastive fine-tuning on product graph |
| **Amazon StyleSnap** | CNN embedding + multi-modal fusion |
| **Google Lens Shopping** | CLIP-family + Shopping Graph |
| **Meta** | CLIP + DINOv2 features combined |

**Recommended for this system:** Start with **CLIP ViT-B/16 fine-tuned on your product catalog**. This gives you:
- Multi-modal search from day 1
- Strong baseline without custom training infrastructure
- Upgrade path to ViT-L/14 when compute budget grows

---

## Interview Checkpoint

1. **"Why not just use pretrained CLIP without fine-tuning?"**
   - Domain gap: CLIP's web pretraining includes general images; product images have different distributions (white backgrounds, consistent angles, specific attributes matter more). Fine-tuning improves Recall@10 by 15–30% on product benchmarks.

2. **"How do you choose embedding dimension (128 vs 512 vs 1024)?"**
   - Larger = more expressive but larger index size and slower ANN search. At 10M products, 512-d float32 = 20GB index. 128-d = 5GB. Trade-off: quality vs. memory. Common choice: 256-d or 512-d.

3. **"What if a user uploads an image with a human wearing the product?"**
   - Need person/background segmentation or object detection to crop the region of interest. Pinterest uses a "lens" that lets users select a crop.

4. **"How do you handle multiple product images per SKU?"**
   - Aggregate: average pooling of embeddings from all images. Or store one embedding per image and aggregate at retrieval. Average pooling is simpler and works well in practice.
