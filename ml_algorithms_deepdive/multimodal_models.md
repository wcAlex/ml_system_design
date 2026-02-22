# Multimodal Models

## 1. What Is Multimodal?

A **multimodal model** processes and reasons over multiple types of data — called **modalities** — within a single model.

```
Unimodal:
  Text  ─────────────────► Language Model ─────────────────► Text output
  Image ─────────────────► Vision Model   ─────────────────► Image output

Multimodal:
  Text  ─────┐
  Image ─────┤─► Multimodal Model ─►  Text / Image / Audio / Any output
  Audio ─────┘
```

**Why does multimodal matter?**

The real world is multimodal. A doctor reads notes AND looks at scans. A customer describes a product AND uploads a photo. A user asks "what is in this chart?" and pastes an image. Unimodal models force users to convert their problem into text — multimodal models meet users where they are.

---

## 2. Modality Types

| Modality | Examples | Common Representation |
|---|---|---|
| **Text** | Articles, code, captions, queries | Token embeddings (Transformer) |
| **Image** | Photos, diagrams, charts, screenshots | Patch embeddings (ViT) or CNN features |
| **Video** | Clips, streams, recordings | Frame-level features + temporal encoding |
| **Audio** | Speech, music, sounds | Spectrogram → CNN or Wav2Vec |
| **Structured data** | Tables, databases, graphs | Row/column embeddings, GNNs |
| **3D / Point cloud** | LiDAR, 3D models, medical volumes | PointNet, voxels |
| **Sensor data** | IMU, GPS, physiological signals | Time-series encoding |

---

## 3. Core Architecture Patterns

### Pattern A: Dual Encoder (Contrastive, like CLIP)

Two separate encoders with no cross-modal attention. Embeddings are aligned by contrastive loss.

```
Image ──► Image Encoder ──► img_embed ─────┐
                                           ├── cosine_sim ──► score
Text  ──►  Text Encoder ──► txt_embed ─────┘
```

**Pros:** Fast — encode each modality once, compare embeddings. Scales to billions of items.
**Cons:** Limited reasoning — can't do "describe the relationship between image and text."
**Use cases:** Image search, cross-modal retrieval, large-scale indexing.

### Pattern B: Cross-Encoder (Fusion, like LLaVA, GPT-4V)

One model ingests all modalities together and generates a response.

```
Image ──► Image Encoder ──► patch tokens ─────┐
                                               ├──► Joint Transformer ──► Text output
Text  ──► Text Tokenizer ──► text tokens  ─────┘
```

**Pros:** Rich reasoning — model attends to both modalities simultaneously.
**Cons:** Slow — must process query + context every time. Can't precompute.
**Use cases:** Visual QA, document understanding, code generation from screenshots.

### Pattern C: Late Fusion

Modality-specific models produce outputs; a fusion layer combines them.

```
Image ──► Vision Model ──► prediction_1 ─────┐
                                              ├──► Fusion (MLP / Attention) ──► Final output
Text  ──► Language Model ──► prediction_2 ───┘
```

**Pros:** Can swap modality-specific components; each can be trained separately.
**Cons:** Fusion happens late — misses fine-grained cross-modal interactions.
**Use cases:** Recommendation systems, medical diagnosis (image + patient notes), autonomous driving (camera + LiDAR).

### Pattern D: Unified Token Sequence (Modern LMM approach)

Convert all modalities to tokens and feed into a single Transformer.

```
[<img> patch_1 patch_2 ... patch_196 </img>] [user: what is in this image?] [assistant: ...]
      │ image tokens treated as text tokens   │  text tokens                 │ generated
```

This is what GPT-4V, LLaVA, Gemini, and Claude do. The image encoder maps visual patches to token-like vectors that the LLM can "read" alongside text.

---

## 4. Key Multimodal Models

### 4.1 CLIP (OpenAI, 2021) — Foundation for Everything After

Aligns image and text embeddings via contrastive pretraining on 400M web pairs.
See `clip_and_contrastive_learning.md` for full details.

**Impact:** CLIP's image encoder became the standard vision backbone for virtually every multimodal model that followed.

---

### 4.2 Flamingo (DeepMind, 2022) — First Large Multimodal LM

**Innovation:** Interleave image embeddings into a frozen large language model without retraining the LLM.

```
Architecture:
  Frozen LLM (Chinchilla 70B)
    ↑ (cross-attention at each layer)
  Perceiver Resampler (compresses variable image patches → fixed 64 tokens)
    ↑
  Frozen CLIP image encoder

Only the cross-attention layers + Perceiver are trained.
```

**Key idea:** The Perceiver Resampler converts any number of image patches into a fixed set of 64 visual tokens. This lets the LLM handle images of different sizes without architecture changes.

**Use case:** Few-shot visual QA, image captioning, visual reasoning.

---

### 4.3 LLaVA (Large Language and Vision Assistant, 2023) — Open-Source Multimodal LM

**Innovation:** Extremely simple architecture — just a linear projection between CLIP and an LLM.

```
Image ──► CLIP ViT-L/14 ──► 256 patch embeddings
                                 │
                            Linear layer (trainable)
                                 │
                            256 visual tokens
                                 │
                    ┌────────────┘
                    ▼
         LLaMA / Vicuna LLM ──► text response

Training:
  Stage 1: Train only the linear projection (align embeddings) — 1 epoch on CC3M captions
  Stage 2: Fine-tune projection + LLM on instruction-following visual QA data
```

**Significance:** Showed that a simple linear adapter (not a complex Perceiver) is enough to get strong visual QA with proper instruction tuning data. The data quality matters more than the architecture complexity.

---

### 4.4 GPT-4V / GPT-4o (OpenAI, 2023–2024)

Architecture details not fully published, but key properties:
- Natively multimodal: handles image, text, and (in 4o) audio in a unified model
- Strong OCR, chart understanding, spatial reasoning
- Can process up to 20 images per request

**4o vs 4V:** GPT-4o ("omni") processes audio directly as tokens, not via a separate speech pipeline. This removes latency of ASR → LLM → TTS.

---

### 4.5 Gemini (Google, 2023) — Natively Multimodal from Pretraining

**Key difference from GPT-4V:** Gemini was designed multimodal from the start — not a vision adapter bolted onto a text model.

```
Pretraining data: text + images + audio + video interleaved
Model: Single Transformer processing all modality tokens
```

**Result:** Better at tasks requiring tight integration across modalities (e.g., understanding a video frame + spoken audio + on-screen text simultaneously).

---

### 4.6 DALL-E / Stable Diffusion — Image Generation from Text

Goes the other direction: text → image.

```
DALL-E 3 (OpenAI):
  Text prompt ──► CLIP text encoder ──► conditioning signal
                                              │
                              Diffusion model (U-Net)
                                              │
                              Noise → denoise → image

Stable Diffusion:
  Text ──► CLIP/T5 encoder ──► cross-attention conditioning
  Latent diffusion: compress image → latent space → denoise in latent → decode
```

**Latent Diffusion Models (LDM):** Key innovation of Stable Diffusion — run the denoising process in a compressed latent space (4× smaller), not pixel space. 10–50x faster than DALL-E 2.

---

### 4.7 Whisper (OpenAI, 2022) — Audio → Text

Large-scale speech recognition via Transformer:

```
Audio waveform ──► log-Mel spectrogram ──► CNN encoder ──► Transformer decoder ──► text
```

Trained on 680K hours of diverse multilingual audio from the web. Zero-shot across 99 languages. Often used as the audio backbone in multimodal pipelines.

---

### 4.8 ImageBind (Meta, 2023) — Six Modalities, One Space

**Innovation:** Bind six modalities into a single shared embedding space using paired data.

```
                    ┌─────────────────┐
Image ─────────────►│                 │
Audio ─────────────►│  Shared Embed   │
Text ──────────────►│  Space (1024-d) │
Depth ─────────────►│                 │
Thermal ───────────►│                 │
IMU ───────────────►│                 │
                    └─────────────────┘

Training trick: only need (image, X) pairs for each modality X.
  Since all modalities are anchored to images, they all become comparable.
  Result: audio ↔ text similarity works even without (audio, text) training pairs.
```

**Use case:** Search across modalities — find a video by humming its audio, or retrieve 3D scenes from text.

---

## 5. Use Cases with Examples

### Use Case 1: Visual Product Search (E-commerce)
**Modalities:** Image + optional text refinement
**Model:** CLIP fine-tuned on product catalog

```
User: [uploads photo of red sneaker] + types "similar but in white"
  │
  ▼
CLIP image encoder → 512-d image embedding
CLIP text encoder  → 512-d text embedding  ("similar but in white")
  │
  ▼
Fused query embedding (0.7 × img + 0.3 × text)
  │
  ▼
FAISS ANN search → Top-500 product results → Re-ranked → Top-20
```

**Companies:** Pinterest, Amazon, Shopify, ASOS

---

### Use Case 2: Document Intelligence
**Modalities:** Image (document scan) + text (OCR'd content + layout)
**Model:** LayoutLM (Microsoft), Donut, GPT-4V

```
Invoice PDF scan ──► Image encoder (patches) + OCR text ──► Joint Transformer
                                                                      │
                                                                      ▼
                                                          Extracted fields:
                                                            {amount: "$432.50",
                                                             date: "2024-03-15",
                                                             vendor: "Acme Corp"}
```

**Companies:** Stripe, Intuit, insurance companies, logistics (bill of lading parsing)

---

### Use Case 3: Medical Report Generation
**Modalities:** Image (X-ray / MRI) + patient history text
**Model:** CheXpert (chest X-ray), BioViL-T (temporal medical)

```
Chest X-ray ──► Medical image encoder (CLIP fine-tuned on radiology)
Patient notes ─►  Medical text encoder
                          │
                     Joint context
                          │
                          ▼
           "Findings: Right lower lobe consolidation consistent
            with pneumonia. No pleural effusion. Cardiomegaly noted."
```

**Impact:** FDA-cleared systems augment radiologist workflow; reduce report turnaround from hours to minutes.

---

### Use Case 4: Video Understanding
**Modalities:** Video frames + audio + subtitles/transcript
**Model:** VideoCLIP, InternVideo, Gemini 1.5

```
YouTube video ──► Sample 32 frames at 1fps
                  │
                  ▼ Per-frame CLIP embeddings + temporal transformer
                  │
                  ▼
Audio transcript (Whisper) ──► text embeddings
                  │
                  ▼
Fused temporal-multimodal representation
                  │
              ┌───┴────────────────┐
              ▼                    ▼
     Video Search              Moment Retrieval
  "Find videos about          "At what timestamp
   hiking in Patagonia"        does the speaker
                               mention Paris?"
```

**Companies:** YouTube (content understanding), TikTok (content policy), Descript (video editing)

---

### Use Case 5: Autonomous Driving Perception
**Modalities:** Camera images + LiDAR point cloud + radar + HD maps
**Model:** BEVFusion, UniAD (NVIDIA), Tesla FSD

```
Camera (8×) ──► ViT image encoders ──► Bird's Eye View (BEV) features
                                                │
LiDAR ────────► PointPillar encoder ───────────┤
                                                │
Radar ────────► Radar encoder ─────────────────┤
                                                │
                                          BEV Fusion
                                                │
                                         3D Object Detection
                                         Lane Prediction
                                         Motion Planning
```

**Key challenge:** Aligning modalities in 3D space despite different coordinate systems and temporal offsets.

---

### Use Case 6: Multimodal Conversational AI (VQA)
**Modalities:** Image(s) + text dialogue
**Model:** LLaVA, GPT-4V, Claude, Gemini

```
User: [screenshot of a bar chart] "Which month had the highest sales?"
  │
  ▼
CLIP image encoder ──► 256 patch tokens ──► Linear projection ──► LLM input
Text: "Which month had the highest sales?" ──► token embedding ──► LLM input
  │
  ▼
LLM generates: "According to the chart, March had the highest sales at $2.3M."
```

**Companies:** Slack (image analysis in messages), Notion AI, Adobe Firefly

---

## 6. Training Data for Multimodal Models

| Data Type | Sources | Scale |
|---|---|---|
| (Image, caption) pairs | LAION, CC12M, WIT, web scrape | Billions |
| (Video, transcript) pairs | YouTube, How2, HowToVQA | Millions |
| (Audio, text) pairs | LibriSpeech, Podcast transcripts | Hundreds of thousands of hours |
| Instruction-following visual QA | LLaVA-Instruct, ShareGPT4V | Hundreds of thousands |
| Interleaved documents | MMC4 (multimodal web docs) | Billions of tokens |

**Critical insight:** Data quality > quantity for fine-tuning. LLaVA's breakthrough was GPT-4-generated instruction-tuning data (158K samples) outperforming models trained on millions of raw captions.

---

## 7. Evaluation

| Task | Benchmark | What It Measures |
|---|---|---|
| Image QA | VQAv2, GQA | Factual visual reasoning |
| OCR / document | TextVQA, DocVQA | Text reading in images |
| Chart understanding | ChartQA | Data visualization comprehension |
| Visual grounding | RefCOCO | Referring to specific regions |
| Image captioning | COCO captions, NoCaps | Caption quality (BLEU, CIDEr) |
| Cross-modal retrieval | MS-COCO retrieval, Flickr30K | Image-text alignment |
| Video QA | MSVD-QA, ActivityNet-QA | Temporal visual reasoning |

---

## 8. Challenges & Open Problems

| Challenge | Details |
|---|---|
| **Hallucination** | Model generates plausible-sounding but wrong facts about the image |
| **Spatial reasoning** | "Left of", "behind", "smaller than" — still difficult |
| **Counting** | Models reliably fail at exact counting (> 5 objects) |
| **Temporal understanding in video** | Long-range dependencies, action causality |
| **Alignment tax** | Fine-tuning for instruction following can degrade retrieval quality |
| **Compute cost** | Cross-encoder models can't be precomputed — each query is expensive |
| **Modality imbalance** | Text-heavy pretraining → text dominates over vision in joint models |

---

## 9. The Big Trend: Everything Is Becoming Multimodal

```
2021: CLIP — align image + text
2022: Flamingo — LLM + vision adapter
2023: GPT-4V — commercial multimodal LLM
2023: LLaVA — open-source multimodal LLM
2023: ImageBind — 6 modalities unified
2024: GPT-4o — image + audio + text natively
2024: Gemini 1.5 — 1M context with video
2025+: ???  — Real-time embodied multimodal agents?
```

The trajectory is clear: future foundation models will be natively multimodal. Unimodal models will be relegated to specialized, latency-critical applications.

---

## 10. Interview Checkpoint

1. **"What's the difference between a dual encoder and a cross-encoder for multimodal tasks?"**
   - Dual encoder: each modality encoded independently, compared by similarity. Fast, scalable, used for retrieval. Cross-encoder: all modalities fed together, allows deep cross-attention. Better reasoning, can't precompute, used for re-ranking and generation.

2. **"How does LLaVA work and why is it significant?"**
   - CLIP image encoder + linear projection + LLaMA. Significant because it showed a simple architecture with good instruction-tuning data matches or beats complex architectures. Data quality > model complexity.

3. **"How do you handle hallucination in multimodal models?"**
   - Grounding: force model to cite regions of the image. RLHF with human feedback on factual accuracy. Retrieval augmentation: ground claims in retrieved evidence. Confidence calibration and uncertainty estimation.

4. **"For a product search system, when do you use a dual encoder vs. a cross-encoder?"**
   - Dual encoder for retrieval over millions of products (precomputed product embeddings). Cross-encoder for re-ranking the top-50 candidates where quality matters more than speed. This two-stage pattern is universal.

5. **"What is the 'alignment tax' in multimodal fine-tuning?"**
   - When you fine-tune a retrieval model (like CLIP) on instruction-following VQA data, its embedding structure changes — retrieval quality degrades even though VQA accuracy improves. Solution: separate models for retrieval and generation, or careful multi-task training with task-specific heads.
