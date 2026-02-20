# Probe Questions: Video Recommendation System

Before jumping into design, you need to ask clarifying questions to scope the problem and uncover constraints. In an interview, this phase demonstrates structured thinking and shows you don't make assumptions.

---

## 1. Business Objective & Success Metrics

- **What does "user engagement" mean specifically?** Watch time? Click-through rate? Number of videos watched per session? Session duration? Return visits?
- **Is there a single primary metric, or a composite?** (e.g., YouTube optimizes for long-term user satisfaction, not just clicks)
- **Are there guardrail metrics?** Things we must NOT degrade -- e.g., user satisfaction surveys, diversity of content consumed, misinformation spread, creator fairness.
- **What is the current baseline?** Are we building from scratch or improving an existing system?
- **What is the time horizon for engagement?** Short-term (this session) vs. long-term (user retention over weeks/months)?

> **Why this matters:** "Increase engagement" is ambiguous. Optimizing for clicks leads to clickbait. Optimizing for watch time leads to rabbit holes. The metric choice fundamentally shapes the model design.

---

## 2. Product Context & User Experience

- **Where does the recommendation appear?** Homepage feed? "Up next" sidebar? End-of-video suggestions? Search results? All of these?
- **How many recommendations do we show at once?** A ranked list of 10? An infinite scroll feed? A single "up next" video?
- **Is this for logged-in users only, or also anonymous/cold-start users?**
- **Are there different user segments?** (new users, power users, creators, etc.)
- **Is there an existing recommendation system we are replacing or augmenting?**

> **Why this matters:** Homepage recs, "up next" recs, and search recs have very different signals and latency requirements. The UX surface determines the problem framing.

---

## 3. Scale & Constraints

- **How many users?** (millions? billions?)
- **How many videos in the catalog?** (millions? hundreds of millions?)
- **How many new videos are uploaded per day?**
- **What is the acceptable latency for serving recommendations?** (e.g., < 200ms)
- **What infrastructure exists?** (GPUs for training, serving fleet, feature store, etc.)
- **What is the QPS (queries per second) we need to support?**

> **Why this matters:** Scale determines whether you need a multi-stage retrieval + ranking pipeline (you almost certainly do at YouTube scale). A 10M video catalog cannot be scored one-by-one per request.

---

## 4. Data Availability

- **What user interaction data do we have?** Impressions, clicks, watch time, likes, dislikes, shares, comments, subscribes, search queries?
- **What video metadata is available?** Title, description, tags, category, upload time, duration, language, creator info?
- **Do we have video content signals?** Thumbnails, audio transcripts, visual embeddings?
- **How much historical data is available?** Days? Months? Years?
- **What is the data freshness requirement?** Real-time streaming or daily batch?
- **Are there data quality issues?** Bot traffic, spam, missing fields?

> **Why this matters:** The richness of your feature set is bounded by available data. If you don't have watch-time data, you can't optimize for watch time. If you have no content embeddings, cold-start is harder.

---

## 5. Cold-Start & Edge Cases

- **How do we handle brand-new users with no history?**
- **How do we handle brand-new videos with no engagement data?**
- **How do we handle users who return after a long absence?**
- **What about niche interests or long-tail content?**

> **Why this matters:** Cold-start is one of the most frequently asked interview questions. Having a clear strategy (popularity-based fallback, content-based features, exploration) is critical.

---

## 6. Content Policy & Fairness

- **Are there content types we must NOT recommend?** (misinformation, harmful content, age-restricted)
- **Are there fairness requirements?** (creator exposure fairness, avoiding filter bubbles)
- **Are there legal/regulatory constraints?** (COPPA for children, GDPR for data usage)
- **Do we need to support parental controls or content ratings?**

> **Why this matters:** In practice, a recommendation system without safety rails is a liability. Interviewers want to see you think beyond pure optimization.

---

## 7. Real-Time vs. Batch

- **Do recommendations need to react in real-time to what the user just watched?**
- **Or is it acceptable to update recommendations periodically (e.g., hourly)?**
- **Do we need to handle trending/breaking content?**

> **Why this matters:** Real-time personalization (reacting to the last 5 minutes of behavior) requires streaming infrastructure and online models, which is a fundamentally different architecture than batch.

---

## 8. Experimentation & Iteration

- **Is there an A/B testing framework in place?**
- **How do we measure offline vs. online performance?**
- **How quickly do we need to iterate on models?**

> **Why this matters:** The gap between offline metrics (AUC, NDCG) and online metrics (actual engagement lift) is a well-known challenge. Your design should account for how you close this gap.

---

## Key Information to Lock Down Before Designing

| Category | Decision | Impact on Design |
|---|---|---|
| Primary metric | e.g., Expected watch time | Shapes the label and loss function |
| Recommendation surface | e.g., Homepage + Up-next | Determines number of models needed |
| Catalog size | e.g., 100M+ videos | Forces multi-stage pipeline (retrieval -> ranking) |
| User base size | e.g., 1B+ users | Dictates infrastructure and embedding table size |
| Latency budget | e.g., < 200ms p99 | Constrains model complexity at serving time |
| Data signals available | e.g., clicks, watch %, likes | Determines feature engineering scope |
| Cold-start prevalence | e.g., 20% new users/day | Requires explicit cold-start strategy |
| Real-time requirement | e.g., Must react within session | Needs streaming features and online updates |
| Safety constraints | e.g., No harmful content | Requires filtering layer and policy model |

---

## Interview Tip

Spend 3-5 minutes on these questions before drawing any architecture. This shows the interviewer you:
1. Don't jump to solutions prematurely
2. Understand that ML system design is driven by requirements, not algorithms
3. Can identify the constraints that actually matter for the design
4. Think about the problem end-to-end (not just the model)

A strong candidate asks ~8-10 targeted questions, locks down the scope, and THEN proceeds to design.
