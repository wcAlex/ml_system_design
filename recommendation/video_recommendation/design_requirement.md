Business Objective: 

Enagement == Customer Satisfaction (per week): 


System Interview Perspective:
1. Development process, interactive process for Production Ready Application
2. Interpretability in ML 
3. Interleaving experiment (Netflix)
4. Chaos testing
5. Training, dataset, where does it come from, how to build one, feature engineering

Satisfaction Score = expression of 
    Click Rate,
    Number of video watched,
    Number of finished video, 
    Watched time,
    likes by user, 


Hybrid model recommendation approach:
Stage 1: CANDIDATE GENERATION (Retrieval)
├── Source A: Collaborative Filtering (two-tower model, ~100 candidates)
├── Source B: Content-Based (similar to recently watched, ~100 candidates)
├── Source C: Popularity (trending videos, ~50 candidates)
├── Source D: Social (what friends watched, ~50 candidates)
├── Source E: Subscriptions (new uploads from followed creators, ~50 candidates)
│
├── Merge & Deduplicate → ~300 unique candidates
│
Stage 2: RANKING
├── A single deep ranking model scores all candidates
├── Uses BOTH CF features (user/item embeddings) AND content features
├── Multi-task: predicts P(click), E[watch_time], P(like), P(finish)
├── Combines into satisfaction score
│
Stage 3: RE-RANKING (Post-Processing)
├── Apply guardrails (diversity, freshness, safety)
├── Final list of ~20-50 videos shown to user


<!-- Explore 

Models: 

MMoE (Multi-gate Mixture of Experts) -- Google 2018
PLE (Progressive Layered Extraction)

| Model                  | Key idea                                      | Limitation                     |
| ---------------------- | --------------------------------------------- | ------------------------------ |
| Hard parameter sharing | One backbone for all tasks                    | Task interference              |
| Single-gate MoE        | One gate for all tasks                        | Same expert mix for every task |
| **MMoE**               | **Per-task gates**                            | Slightly more compute          |
| PLE                    | Hierarchical experts (shared + task-specific) | More complex                   |


my open questions:
1. Knn algorihtm 
2. Difference between recommendation and search systems.  -->