# Vision Transformer (ViT) & Self-Supervised ViT (DINO / DINOv2)

## Part 1: What Is a Vision Transformer (ViT)?

### The Problem ViT Solves

CNNs process images through small local kernels (e.g., 3×3 filters). A pixel at the top-left of an image can only "talk to" its immediate neighbors in early layers. To connect distant parts of the image, you need many stacked layers.

```
CNN: information travels slowly across the image
  ┌───────────────────────────────────────┐
  │  ■ ■ ■ · · · · · · · · · · · · · ·  │
  │  ■ ■ ■ · · · · · · · · · · · · · ·  │  These pixels can't
  │  ■ ■ ■ · · · · · · · · · · · · · ·  │  "see" each other in
  │  · · · · · · · · · · · · ■ ■ ■ · ·  │  early layers
  │  · · · · · · · · · · · · ■ ■ ■ · ·  │
  └───────────────────────────────────────┘

ViT: every patch can attend to every other patch in layer 1
  ┌───────────────────────────────────────┐
  │  ■ ■ ■ · · · · · · · · · · · · · ·  │
  │  ■ ■ ■ · · · ←───────────────→ ■ ■ ■│  Direct attention
  │  ■ ■ ■ · · · · · · · · · · · · · ·  │  from layer 1
  └───────────────────────────────────────┘
```

ViT (Dosovitskiy et al., 2020) borrowed the Transformer architecture from NLP and applied it directly to images by treating image patches as "words."

---

### Step-by-Step: How ViT Works

#### Step 1: Split the Image Into Patches

```
Input image: 224 × 224 pixels
Patch size: 16 × 16 pixels

Number of patches: (224/16) × (224/16) = 14 × 14 = 196 patches

┌────────────────────────────────────────────┐
│ P1  │ P2  │ P3  │ P4  │ ... │ P14 │
├────────────────────────────────────────────┤
│ P15 │ P16 │ P17 │ P18 │ ... │ P28 │
├────────────────────────────────────────────┤
│ ... │     │     │     │     │     │
├────────────────────────────────────────────┤
│P183 │P184 │ ... │     │     │ P196│
└────────────────────────────────────────────┘

Each patch Pi is a 16×16×3 = 768 raw pixel values.
```

#### Step 2: Flatten Each Patch into a Vector (Patch Embedding)

```
Each patch: 16 × 16 × 3 = 768 values
                │
        Linear projection (768 → 768-d)   ← a learned weight matrix
                │
        768-d patch embedding token

Think of each patch as a "word" in a sentence.
196 patches → 196 "words" (tokens)
```

#### Step 3: Add a Special [CLS] Token + Positional Encoding

```
[CLS] P1  P2  P3  ... P196
  ↑
A learnable token prepended to the sequence.
It has no image content — it will accumulate a summary
of the whole image through attention.

Positional encoding added to each token so the model
knows "patch 15 is in row 2, column 1."

Sequence length: 1 (CLS) + 196 (patches) = 197 tokens
```

#### Step 4: Feed Through Transformer Encoder (Self-Attention Layers)

```
Input: 197 tokens × 768-d

Each layer applies Multi-Head Self-Attention:
  - Every token "looks at" every other token
  - Learns which patches are related to which

Example for a shoe image:
  Patch at the sole attends strongly to → patches at the toe, heel
  Patch at the laces attends to        → patches at the tongue, eyelets
  [CLS] token attends to               → all patches (building a global summary)

After 12 layers (ViT-B) of this: [CLS] token contains a rich
representation of the entire image's content and structure.
```

#### Step 5: Extract Embedding from [CLS] Token

```
After 12 Transformer layers:

[CLS] ← 768-d vector representing the whole image
 P1   ← 768-d vector (patch 1's refined features)
 P2   ← 768-d vector
 ...
 P196 ← 768-d vector

For image classification / retrieval:
  → Take [CLS] token
  → Pass through a small linear layer → 512-d embedding
  → L2 normalize → done

For dense prediction (segmentation, depth):
  → Use all 196 patch tokens (not just [CLS])
  → Each patch token corresponds to a 16×16 region of the image
```

#### Full ViT Architecture Summary

```
224×224 Image
     │
     ▼ Split into 196 patches of 16×16
     │
     ▼ Linear Projection → 196 × 768-d tokens
     │
     ▼ Prepend [CLS] → 197 × 768-d
     │
     ▼ Add Positional Encoding
     │
     ▼ Transformer Encoder (12 layers for ViT-B)
       Each layer:
         ├─ Multi-Head Self-Attention (every token ↔ every token)
         └─ Feed-Forward Network (FFN)
     │
     ▼ [CLS] token output: 768-d
     │
     ▼ Linear projection → 512-d
     │
     ▼ L2 normalize → final embedding
```

---

### ViT vs. CNN: Key Differences

| Dimension | CNN (ResNet) | ViT |
|---|---|---|
| **Reception field** | Local (3×3 kernel) | Global (self-attention in layer 1) |
| **How it sees context** | Builds up slowly through many layers | Sees full context from the start |
| **Inductive biases** | Strong: locality, translation equivariance | Weak: must learn spatial structure from data |
| **Data requirements** | Good with small datasets | Needs large datasets (or pretraining) |
| **Fine-grained textures** | Strong (local filters are great for texture) | Weaker than CNN without extra help |
| **Global structure** | Weak until deep layers | Strong from layer 1 |
| **Scalability** | Hits a quality ceiling at large scale | Keeps improving with more data and compute |

**Practical implication:** ViT is better when you need to understand the global structure of an image (whole outfit, scene composition). CNN is better when local texture matters (fabric pattern, stitching detail) or you have limited training data.

---

## Part 2: Self-Supervised Learning — The "No Labels" Approach

Before diving into DINO, you need to understand why self-supervised learning exists.

### The Labeling Problem

Supervised ViT training needs millions of labeled images:

```
Traditional supervised training:
  Image of sneaker → human says "sneaker" → model learns
  Image of dress   → human says "dress"   → model learns

Problems:
  - Labeling is expensive ($2–5 per image)
  - Labels tell the model the category, not the visual structure
  - You can't label "this patch looks like that patch"
  - Labels are limited to predefined classes — model can't generalize beyond them
```

### The Self-Supervised Insight

What if the model could generate its own learning signal from the structure of the data itself?

```
Self-supervised learning:
  Image of sneaker → model must predict something about the image
                     from the image itself (no human labels)

Examples of self-supervised "pretext tasks":
  - Predict the rotation of the image (0°, 90°, 180°, 270°)
  - Predict a missing patch (like BERT's masked token prediction)
  - Predict if two crops of the same image are from the same image
```

The model doesn't learn "this is a sneaker" — it learns rich visual representations by solving these structural puzzles. Those representations transfer to downstream tasks (retrieval, classification, detection).

---

## Part 3: DINO — Self-Supervised ViT, Step by Step

**DINO** = **Di**stillation with **No** labels (Meta AI, 2021)

### The Core Idea in One Sentence

> Take one image, create two differently-augmented views of it, and train a "student" network to produce the same output as a "teacher" network — where the teacher is a slowly-updated copy of the student.

### Step-by-Step Walkthrough

#### Step 1: Take One Image, Create Multiple Views

```
Original image: a sneaker

           ┌─────────────────┐
           │   Augmentation  │
           └────────┬────────┘
           │                 │
           ▼                 ▼
Global crop 1         Global crop 2
(large crop,          (large crop,
 different area,       different color)
 flipped)

           +
Local crops 3–8 (small patches of the image)

Why "global" vs "local"?
  Teacher sees GLOBAL crops (large context)
  Student sees ALL crops (global + local)
  → Forces student to predict global context from local detail
  → Learns "this small patch belongs to the larger sneaker shape"
```

#### Step 2: Two Networks — Student and Teacher

```
Same Architecture (ViT-S, ViT-B, etc.)
Same Initial Weights

  Student Network                 Teacher Network
  ──────────────                  ────────────────
  Receives all crops              Receives only global crops
  (global + local)
        │                               │
        ▼                               ▼
  Student output                  Teacher output
  (probability distribution       (probability distribution
   over K=65536 "concepts")        over same K concepts)
        │                               │
        └──────── Loss: make these ─────┘
                  distributions match
```

The K=65536 "concepts" are not predefined categories. They emerge during training — the model discovers its own visual vocabulary.

#### Step 3: The Teacher Updates Slowly (EMA)

This is the most important part. The teacher is **NOT trained by gradient descent**. Instead:

```
After each training step:

Teacher weights = 0.9996 × (old Teacher weights)
               + 0.0004 × (current Student weights)

This is called Exponential Moving Average (EMA).

Why?
  - If teacher = student: they'd find a trivial solution (both output the same constant)
  - Teacher changes SLOWLY → provides a stable training target
  - Teacher is like a "slightly older, more conservative version" of the student
  - Over time, the teacher is an ensemble of many past student versions
    → more stable, higher quality than the student alone

Analogy: Imagine you're learning to cook (student).
Your teacher is your past self from 2 weeks ago (teacher).
You try to match what your past self would have done,
but you're always slightly ahead.
```

#### Step 4: The Loss Function

```
For each image:
  - Teacher produces a probability distribution: [0.3, 0.0, 0.5, 0.2, ...]
  - Student produces a probability distribution: [0.2, 0.1, 0.6, 0.1, ...]

Loss = cross-entropy(teacher_output, student_output)
     = -Σ teacher_prob × log(student_prob)

Goal: Student output should match Teacher output
      (distributions should be as close as possible)
```

But there's a **collapse problem**: the easiest solution is for both student and teacher to output the same thing for ALL images — e.g., always output [1, 0, 0, 0, ...]. That satisfies the loss trivially, but is useless.

#### Step 5: Preventing Collapse — Centering and Sharpening

```
Two tricks to prevent collapse:

1. CENTERING (applied to teacher output):
   Subtract the running mean of teacher outputs
   → Prevents one dimension from dominating
   → Forces the distribution to be spread out across many concepts

2. SHARPENING (via temperature τ):
   Teacher: divide logits by a small temperature (τ_teacher = 0.04)
   → Makes teacher output SHARP (high confidence)
   → Forces student to make confident predictions too
   → Student can't "cheat" by outputting a flat distribution

student_input: local crop (small piece of sneaker)
teacher_target: SHARP distribution over global concept
→ Student must figure out: "what global concept does this local patch belong to?"
→ This is hard, so the model must learn real visual structure
```

#### Step 6: What DINO Actually Learns

After training, something remarkable happens — the model learns to segment objects without any segmentation labels:

```
Input: photo of a dog in a park

DINO attention map (what the [CLS] token attends to):

Before training:
  ┌─────────────────────┐
  │ · · · · · · · · · · │  Attention spread everywhere
  │ · · · · · · · · · · │  (random)
  │ · · · · · · · · · · │
  └─────────────────────┘

After DINO self-supervised training:
  ┌─────────────────────┐
  │ · · ██████████ · · │  Attention focuses on the dog!
  │ · ████████████ · · │  (no segmentation labels used)
  │ · · ███ · · · · · │
  └─────────────────────┘
```

**Why?** To predict that a local crop (e.g., a paw) belongs to the same global concept as another crop (e.g., a snout), the model must understand that both are part of the same object. It implicitly learns object boundaries.

---

### DINO vs. DINOv2

| Aspect | DINO (2021) | DINOv2 (2023) |
|---|---|---|
| **Training data** | ImageNet-1M (uncurated) | LVD-142M (142M curated web images) |
| **Data curation** | None | Self-supervised deduplication + quality filtering |
| **Additional losses** | Self-distillation only | + iBOT (masked image modeling) + SwAV |
| **Backbone** | ViT-S, ViT-B | ViT-S, ViT-B, ViT-L, ViT-g (giant) |
| **Quality** | Strong | State-of-the-art (beats supervised on many tasks) |
| **Dense features** | Good | Excellent (patch tokens are very informative) |

**DINOv2's key addition — iBOT (image BERT):**
```
Random patches are MASKED during training.
Model must predict the masked patch's representation
(like BERT's [MASK] token prediction, but for images).

This forces the model to understand each patch's content
in context — great for dense tasks (segmentation, depth, detection).
```

---

### Concrete Example: DINO for Shoe Retrieval

```
Training (self-supervised, no labels):

  Image of shoe (full photo)
       │
       ├──► Global crop 1: whole shoe, slight rotation
       ├──► Global crop 2: whole shoe, color jitter
       ├──► Local crop 3: just the sole (tiny patch)
       ├──► Local crop 4: just the laces (tiny patch)
       └──► Local crop 5: just the toe box (tiny patch)

  Teacher (global crops) → output: [0.0, 0.0, 0.8, 0.0, 0.2, ...]
                                            ↑ concept 3 = "this sneaker style"

  Student (local crop: sole) → must output: [0.0, 0.0, 0.8, 0.0, 0.2, ...]
  ← How does the model do this?
  ← It must understand: "this rubber sole patch is PART of a sneaker"
  ← It learns the global concept from a local view

After training:
  Query image: photo of same sneaker model, different angle
       │
       ▼ DINOv2 ViT-B/14
       │
  512-d embedding: [0.23, -0.11, 0.87, ...]

  vs. Catalog sneaker (same model): cosine_sim = 0.96  ← very similar
  vs. Different sneaker:            cosine_sim = 0.71  ← less similar
  vs. Handbag:                      cosine_sim = 0.12  ← very different
```

---

## Part 4: DINO vs. CLIP — What's the Difference?

This is a common point of confusion since both use ViT and produce embeddings.

```
                    CLIP                        DINO/DINOv2
                    ────                        ───────────
Training signal:    (image, text caption) pairs  Image alone (no text)
What it aligns:     Image ↔ Text                 Image views ↔ Image views
Knowledge source:   Web captions (semantic)      Visual structure (pixel-level)

Strengths:
  Text → image queries ✓              Fine-grained visual detail ✓
  Zero-shot classification ✓          Dense prediction (segmentation) ✓
  Broad semantic understanding ✓      Exact instance retrieval ✓

Weaknesses:
  Fine-grained texture/detail ✗       No text understanding ✗
  Dense features less rich ✗          Needs visual queries only ✗

Best for:
  Multimodal search, classification   Pure image similarity, dense tasks
```

**Practical rule:**
- Need to search with text OR classify zero-shot? → **CLIP**
- Need best pure image-to-image similarity, dense features? → **DINOv2**
- Want both? → **Combine them**: concatenate CLIP + DINOv2 embeddings, or use them in ensemble

---

## Part 5: Why "No Labels Required" Matters in Practice

```
Scenario: You're building search for a niche furniture startup.
  - 50,000 product images (chairs, tables, lamps)
  - No labels, no click data, zero user history
  - Small team, no ML labeling budget

Option A: Supervised CNN (ResNet)
  - Need labeled categories → expensive labeling project
  - Only learns predefined categories (can't generalize to "mid-century modern style")

Option B: CLIP (zero-shot)
  - Works out of the box (no fine-tuning needed)
  - Text queries work: "teak dining table with hairpin legs"
  - But: web-pretrained may not know your specific product catalog

Option C: DINOv2 (self-supervised)
  - No labels needed — train directly on your 50K product images
  - Learns YOUR product visual distribution (what makes your chairs distinctive)
  - Image-to-image search: user uploads a chair photo → finds visually identical chairs
  - Doesn't require text input

Best approach: Use pretrained DINOv2 as a feature extractor (no training),
  then fine-tune lightly on your catalog with contrastive pairs (same-SKU = positive).
  No labels required beyond "these two photos are of the same product."
```

---

## Part 6: Summary — The Full Mental Model

```
ViT (Architecture):
  "Treat image patches as words. Use Transformer self-attention
   so every patch can directly attend to every other patch.
   [CLS] token accumulates the global image summary."

Self-Supervised Learning (Training Paradigm):
  "Generate your own supervision from the data structure.
   No human labels — the model learns by solving puzzles
   about the data itself."

DINO (Specific Method):
  "Student tries to match Teacher.
   Teacher = slow EMA of Student → stable target.
   Student sees local crops, Teacher sees global crops.
   Forces model to understand: 'what whole object does this piece belong to?'
   Result: model learns object-level semantics with zero labels."

DINOv2 (Production-Ready DINO):
  "DINO + better data curation + masked image modeling.
   Produces state-of-the-art features for image retrieval,
   segmentation, and depth estimation."
```

---

## Interview Checkpoint

1. **"What's the key difference between a CNN and a ViT?"**
   - CNN: local receptive field, builds global context slowly through many layers. ViT: self-attention lets every patch see every other patch from layer 1. ViT captures global structure better; CNN captures local texture better.

2. **"In DINO, why does the teacher use EMA instead of being trained by gradient descent?"**
   - If teacher = student (identical), they'd collapse to a trivial solution (output constant). EMA makes the teacher a stable, slowly-changing target — like a moving average of recent student checkpoints — which prevents collapse while still providing meaningful learning signal.

3. **"How does DINO learn to segment objects without segmentation labels?"**
   - To predict that a local crop (a paw) belongs to the same global concept as another crop (a snout), the model must understand object boundaries. This object-level grouping emerges from the local-to-global prediction task, giving DINO's attention maps emergent segmentation properties.

4. **"When would you pick DINOv2 over CLIP for image search?"**
   - DINOv2 when: queries are always images (no text), you need fine-grained visual similarity (exact product match, texture), or dense patch features matter (local region matching). CLIP when: you need text + image queries, or zero-shot classification over unseen categories.

5. **"What is catastrophic collapse in self-supervised learning and how does DINO prevent it?"**
   - Collapse = model outputs the same embedding for all inputs (satisfies the loss trivially). DINO prevents it with (1) centering — subtract running mean of teacher outputs to spread the distribution, and (2) sharpening — low temperature on teacher output forces confident, non-uniform predictions.
