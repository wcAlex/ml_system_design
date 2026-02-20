# Data Collection for Video Recommendation Pipeline

This document maps every piece of data we need to collect, WHY we need it, and WHICH stage of the pipeline consumes it.

---

## Data Overview

```
Data Sources
├── 1. User Interaction Logs (implicit + explicit feedback)
├── 2. User Profile Data
├── 3. Video Metadata
├── 4. Video Content Signals
├── 5. Social / Network Data
├── 6. Context Data
└── 7. Negative / Safety Signals
```

---

## 1. User Interaction Logs

This is the most critical data source. Every user action on the platform is logged as an event.

### Implicit Feedback (user didn't intentionally give a signal -- we infer it)

| Event | Fields to Log | Why We Need It |
|---|---|---|
| **Impression** | user_id, video_id, position, timestamp, surface (homepage/sidebar/search), device | Denominator for CTR. Without impressions, we can't tell "user saw it and ignored it" from "user never saw it" |
| **Click** | user_id, video_id, timestamp, source_surface, source_position | Numerator for CTR. Label for click prediction model |
| **Video start** | user_id, video_id, timestamp | Distinguishes "clicked thumbnail but bounced" from "actually started watching" |
| **Watch time** | user_id, video_id, watch_duration_sec, video_duration_sec, timestamp | Primary engagement signal. Also gives us completion rate (watch_duration / video_duration) |
| **Watch percentage checkpoints** | user_id, video_id, percent_watched (25%, 50%, 75%, 100%), timestamp | More granular than total watch time. "Finished video" = 100% checkpoint |
| **Pause / Resume** | user_id, video_id, pause_timestamp, resume_timestamp | Signals attention level. Long pause + resume = strong interest. Pause + never resume = lost interest |
| **Seek (skip forward/back)** | user_id, video_id, seek_from_sec, seek_to_sec, timestamp | Skip forward = boring section. Seek back = rewatching (high interest) |
| **Session data** | user_id, session_id, session_start, session_end, videos_watched[] | Needed for session-level features ("what did the user watch before this?") and session duration metrics |
| **Search queries** | user_id, query_text, timestamp, clicked_results[] | Reveals explicit intent. Bridges search and recommendation |
| **Scroll / dwell time** | user_id, video_id, dwell_time_on_thumbnail_ms, scrolled_past (bool) | Dwell time on thumbnail without clicking = weak negative signal |

### Explicit Feedback (user intentionally gave a signal)

| Event | Fields to Log | Why We Need It |
|---|---|---|
| **Like** | user_id, video_id, timestamp | Strong positive signal. Label for like prediction model |
| **Dislike** | user_id, video_id, timestamp | Strong negative signal. Used in satisfaction score (negative weight) and as guardrail |
| **Comment** | user_id, video_id, comment_text, timestamp | Engagement signal. Sentiment of comment is also useful |
| **Share** | user_id, video_id, share_target (platform/DM), timestamp | Very strong positive signal -- user is recommending it to others |
| **Subscribe (to creator)** | user_id, creator_id, timestamp, source_video_id | Very strong signal of creator affinity. Powers subscription-based retrieval (Source E) |
| **Add to playlist / Save** | user_id, video_id, playlist_id, timestamp | Intentional save = high interest, especially for "watch later" |
| **"Not interested"** | user_id, video_id, timestamp | Explicit negative. Should be hard-respected (never show again) |
| **Report** | user_id, video_id, reason, timestamp | Safety signal. Powers guardrail filtering |

### Why Both Implicit and Explicit Matter

```
Implicit signals:  High volume, noisy, always available
                   (every user generates watch time data)

Explicit signals:  Low volume, high quality, sparse
                   (only ~1-5% of users ever click "like")

You need both. Implicit data trains the bulk of the model.
Explicit data provides strong supervision for specific objectives.
```

---

## 2. User Profile Data

Relatively static attributes about the user.

| Field | Source | Why We Need It |
|---|---|---|
| user_id | Registration | Primary key |
| age / age_group | Registration | Content appropriateness, interest patterns differ by age |
| gender | Registration (optional) | May correlate with content preferences |
| country / region | Registration or IP geolocation | Language, cultural relevance, legal constraints (GDPR, COPPA) |
| language preference | Settings | Filter and rank content in preferred language |
| account age | Registration date | Distinguishes new vs. established users (cold-start handling) |
| device type | Client telemetry | Mobile users prefer shorter videos. TV users prefer longer content |
| subscription list | User actions | Powers Source E (subscription-based retrieval) |
| notification settings | User settings | Indicates engagement level and preferred creator set |

### Derived User Features (computed, not raw)

| Feature | Derived From | Why |
|---|---|---|
| Avg watch time per session | Interaction logs | User-level engagement baseline |
| Preferred video duration | Watch history | Short-form vs. long-form preference |
| Preferred categories/topics | Watch history | Topic affinity vector |
| Activity pattern (time of day) | Interaction timestamps | "This user watches at night" → different content fits |
| Engagement level (power user vs. casual) | Interaction frequency | Adjusts exploration rate |

---

## 3. Video Metadata

Attributes about the video itself, mostly available at upload time.

| Field | Source | Why We Need It |
|---|---|---|
| video_id | System-generated | Primary key |
| title | Uploader | Text features for content-based filtering. Also clickbait detection |
| description | Uploader | Richer text signal for topic extraction |
| tags / categories | Uploader + auto-classification | Powers content-based retrieval (Source B), diversity re-ranking |
| duration_seconds | Video processing | Critical feature: users have duration preferences |
| upload_timestamp | System | Freshness signal. Age of video = time since upload |
| creator_id | Uploader | Creator affinity features |
| language | Auto-detection or uploader | Language matching with user preference |
| thumbnail_url | Uploader | Input to visual models for content understanding |
| resolution / quality | Video processing | Quality signal |
| has_subtitles | Video processing | Accessibility and language reach |
| is_age_restricted | Policy classifier or uploader | Guardrail filtering |
| content_category | Auto-classification model | Topic-level features |

### Derived Video Features (computed after upload)

| Feature | Derived From | Why |
|---|---|---|
| Total views | Interaction logs (aggregated) | Popularity signal for Source C |
| View velocity (views/hour) | Interaction logs (windowed) | Trending detection for Source C |
| Avg watch percentage | Watch time logs | Quality signal -- do people actually finish it? |
| Like/dislike ratio | Explicit feedback | Quality signal, guardrail input |
| CTR (click-through rate) | Impressions + clicks | How attractive is the thumbnail/title combo? |
| Avg watch time | Watch time logs | Key feature for watch time prediction model |
| Comment count / rate | Comment logs | Engagement/controversy signal |
| Share rate | Share logs | Virality signal |
| Creator subscriber count | Subscription data | Creator authority signal |

---

## 4. Video Content Signals

Extracted from the video content itself using ML models. These power content-based retrieval (Source B) and cold-start handling.

| Signal | How It's Extracted | Why We Need It |
|---|---|---|
| **Visual embeddings** | Frame sampling → CNN (e.g., ResNet, ViT) → embedding vector | Capture visual style, scenes, objects. Two visually similar videos get similar embeddings |
| **Audio/transcript** | Speech-to-text (Whisper) → text | Topic understanding from spoken content |
| **Text embeddings** | Title + description + transcript → language model (BERT) → embedding vector | Semantic understanding of video content. Powers "similar to recently watched" |
| **Topic distribution** | NLP topic model on text features | Soft category assignment (a video can be 60% tech, 30% comedy) |
| **Thumbnail embedding** | Image model on thumbnail | Captures visual appeal and content hint |

```
These are expensive to compute but only need to run ONCE per video at upload time.
Store them in a feature store / embedding index for retrieval.
```

---

## 5. Social / Network Data

Powers Source D (social recommendations).

| Field | Source | Why We Need It |
|---|---|---|
| Friend/follow graph | Social features on platform | "Friends who watched this" |
| Shared videos | Share events | "Your friend shared this with you" |
| Co-watch patterns | Session logs from same household/IP | Household interest modeling |
| Creator-subscriber graph | Subscription data | Community detection, collaborative signals |

**Note:** Not all platforms have social features. If unavailable, Source D is dropped and other sources compensate.

---

## 6. Context Data

Captured at REQUEST TIME (when the user opens the app and we need to generate recommendations).

| Field | Source | Why We Need It |
|---|---|---|
| Current timestamp | System clock | Time-of-day features (morning vs. night content) |
| Day of week | System clock | Weekend vs. weekday behavior differs |
| Device type | Client | Mobile → shorter videos, TV → longer videos |
| Connection speed | Client | Affects video quality recommendations |
| Current page/surface | Client | Homepage vs. watch-next have different intent |
| Last N videos watched (this session) | Session log | Real-time session context. The most recent signal available |
| Referring source | Client | Came from search? notification? external link? |

**Why context matters:** The same user wants different content at 8am on a Monday (quick news) vs. 9pm on a Saturday (long documentary). Context features capture this.

---

## 7. Negative / Safety Signals

Powers guardrail filtering and quality controls.

| Signal | Source | Why We Need It |
|---|---|---|
| Policy violation flags | Content moderation classifier | Hard filter: remove from all candidate pools |
| Spam/bot flags | Anti-abuse system | Exclude from engagement metrics (don't let bots inflate trending) |
| Copyright claims | Rights management system | May need to suppress or remove |
| User block list | User actions ("block creator") | Hard filter per-user: never recommend this creator |
| "Not interested" signals | User actions | Hard filter per-user: never recommend this video/topic |
| Report history | User reports | Affects quality score, triggers review |

---

## Data → Pipeline Stage Mapping

Shows which data feeds which stage of the recommendation pipeline.

```
STAGE 1: CANDIDATE GENERATION (Retrieval)
┌─────────────────────────────────────────────────────────────────┐
│ Source A (CF two-tower):                                        │
│   Needs: user interaction logs (watch, click, like history)     │
│   → trains user & video embeddings                              │
│                                                                 │
│ Source B (Content-based):                                       │
│   Needs: video content signals (text/visual embeddings),        │
│          user's recent watch history                             │
│   → finds videos with similar embeddings to recently watched    │
│                                                                 │
│ Source C (Popularity):                                          │
│   Needs: aggregated view counts, view velocity, trending scores │
│   → returns globally/regionally trending videos                 │
│                                                                 │
│ Source D (Social):                                              │
│   Needs: social graph, friends' interaction logs                │
│   → returns videos friends engaged with                         │
│                                                                 │
│ Source E (Subscriptions):                                       │
│   Needs: user subscription list, video upload timestamps        │
│   → returns recent uploads from subscribed creators             │
└─────────────────────────────────────────────────────────────────┘

STAGE 2: RANKING
┌─────────────────────────────────────────────────────────────────┐
│ Deep ranking model needs ALL of the following as features:      │
│   - User profile (age, country, device, preferences)            │
│   - User behavioral features (avg watch time, activity pattern) │
│   - Video metadata (duration, age, category, creator stats)     │
│   - Video quality signals (avg completion rate, like ratio)     │
│   - CF embeddings (user embedding, video embedding from Stage1) │
│   - Content embeddings (text/visual from video)                 │
│   - Context (time of day, device, session history)              │
│   - Cross features (user-video interaction history if any)      │
│                                                                 │
│ Training labels (from interaction logs):                        │
│   - clicked (binary)                                            │
│   - watch_time (continuous, in seconds)                         │
│   - liked (binary)                                              │
│   - finished (binary, watch_pct >= threshold)                   │
└─────────────────────────────────────────────────────────────────┘

STAGE 3: RE-RANKING
┌─────────────────────────────────────────────────────────────────┐
│ Guardrail inputs:                                               │
│   - Video categories/topics (for diversity enforcement)         │
│   - Video upload timestamp (for freshness rules)                │
│   - Safety/policy flags (for hard filtering)                    │
│   - Creator IDs (for creator-level diversity caps)              │
│   - User's "not interested" / block list                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Training Labels: How Interaction Logs Become Supervision

This is worth calling out because it's a common interview question: "How do you get labels?"

| Model | Label | Source | Type |
|---|---|---|---|
| Click prediction | 1 if clicked, 0 if impressed but not clicked | Impression + click logs | Binary classification |
| Watch time prediction | watch_duration_seconds | Watch time logs | Regression |
| Like prediction | 1 if liked, 0 if watched but not liked | Like logs + watch logs | Binary classification |
| Completion prediction | 1 if watch_pct >= 80%, 0 otherwise | Watch time + video duration | Binary classification |

**Critical nuance -- selection bias:** The watch time and like models are trained only on CLICKED videos (you can't watch something you didn't click). This means they predict `E[watch_time | clicked]` and `P(like | clicked)`, not unconditional. This is correct and intentional -- at serving time, these predictions are combined with `P(click)` in the satisfaction score.

---

## Data Infrastructure Summary

| Component | What It Stores | Access Pattern |
|---|---|---|
| **Event log / streaming** (Kafka) | Raw interaction events in real-time | Write-heavy, append-only |
| **Data warehouse** (BigQuery/Hive) | Historical logs for batch training | Batch reads for model training |
| **Feature store** (Feast/Tecton) | Pre-computed features for users & videos | Low-latency reads at serving time |
| **Embedding index** (FAISS/ScaNN) | Video embeddings for ANN retrieval | Sub-millisecond nearest-neighbor search |
| **User profile store** (Redis/DynamoDB) | Latest user state and recent history | Low-latency key-value lookup |
| **Video metadata store** (DB/cache) | Video attributes and derived stats | Low-latency lookup by video_id |

---

## Interview Talking Points

1. **"Impressions are as important as clicks"** -- Without impression logs, you can't distinguish "not interested" from "not shown." This creates position bias if ignored.

2. **"Implicit vs. explicit feedback trade-off"** -- Implicit is abundant but noisy (did they watch because they liked it, or because they fell asleep?). Explicit is clean but sparse (< 5% of users click like).

3. **"Selection bias in training data"** -- You only observe watch time for clicked videos. Models must be aware of this. Solutions: inverse propensity weighting, or separate click and post-click models (which is what the multi-task approach does).

4. **"Feature freshness matters"** -- A user's "last 10 videos watched" changes every minute. Stale features = stale recommendations. This is why you need a real-time feature store, not just a batch pipeline.
