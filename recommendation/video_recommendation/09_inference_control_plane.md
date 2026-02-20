# Model Inference Control Plane Design

The control plane is the brain behind model serving. It decides **which model** serves **which traffic** with **what configuration** — while the data plane handles the actual prediction requests.

---

## 1. Control Plane vs. Data Plane

```
Control Plane (this doc):                   Data Plane:
  "WHICH model, WHERE, HOW"                  "Process THIS request, return prediction"
  ─────────────────────────                   ─────────────────────────────────────────
  Model registry & versioning                 Load model weights into GPU
  Traffic routing (A/B, canary)               Feature lookup from feature store
  Deployment orchestration                    Forward pass through model
  Rollback decisions                          Return prediction scores
  Health monitoring                           Latency-sensitive, high-throughput
  Resource allocation (GPU scheduling)
  Configuration management

Analogy: Air traffic control vs. the airplane
  Control plane = air traffic control tower (routing, scheduling, safety)
  Data plane = the airplane itself (actually flies passengers)
```

---

## 2. Why Do We Need a Control Plane?

Without a control plane, you have a single model hardcoded in production. Any change requires redeployment. With a control plane:

```
Problem                              Control Plane Solution
───────                              ─────────────────────
"We trained a better model"          → Deploy new version to 5% traffic (canary)
"New model increased dislikes"       → Automatic rollback to previous version
"We want to test 3 ranking formulas" → A/B test with traffic splitting
"GPU is overloaded"                  → Auto-scale or degrade gracefully
"Model serving is slow"              → Route to shadow model, monitor latency
"Business wants to boost new content"→ Update ranking weights without redeploying
"Holiday traffic spike"              → Pre-scale GPU instances
"Model is stale (not retrained)"     → Alert, auto-trigger retraining pipeline
```

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CONTROL PLANE                                 │
│                                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ Model        │  │ Traffic      │  │ Config       │               │
│  │ Registry     │  │ Router       │  │ Manager      │               │
│  │              │  │              │  │              │               │
│  │ • Versions   │  │ • A/B tests  │  │ • Weights    │               │
│  │ • Artifacts  │  │ • Canary     │  │ • Thresholds │               │
│  │ • Metadata   │  │ • Shadow     │  │ • Flags      │               │
│  │ • Lineage    │  │ • Rollout %  │  │ • Overrides  │               │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘               │
│         │                  │                  │                       │
│  ┌──────┴──────────────────┴──────────────────┴───────┐              │
│  │              Deployment Orchestrator                │              │
│  │   • Deploy model versions to serving fleet          │              │
│  │   • Manage GPU allocation                           │              │
│  │   • Handle rolling updates                          │              │
│  │   • Execute rollback                                │              │
│  └──────────────────────┬─────────────────────────────┘              │
│                          │                                            │
│  ┌───────────────────────┴────────────────────────────┐              │
│  │              Health Monitor                         │              │
│  │   • Latency p50/p99 tracking                        │              │
│  │   • Error rate monitoring                           │              │
│  │   • Prediction quality metrics                      │              │
│  │   • Auto-rollback triggers                          │              │
│  └────────────────────────────────────────────────────┘              │
│                                                                       │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                    Control signals (which model, what config)
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA PLANE                                   │
│                                                                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │ GPU Pod 1  │  │ GPU Pod 2  │  │ GPU Pod 3  │  │ GPU Pod N  │    │
│  │ Model v12  │  │ Model v12  │  │ Model v13  │  │ Model v12  │    │
│  │ (95%)      │  │ (95%)      │  │ (5% canary)│  │ (95%)      │    │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘    │
│                                                                       │
│  API Gateway → Feature Store → Model Inference → Response             │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Core Components

### 4.1 Model Registry

The source of truth for all model versions and their metadata.

```
Model Registry stores:

  model_name:        "ranking_mmoe_v13"
  model_version:     13
  artifact_path:     "s3://models/ranking/mmoe/v13/model.pt"
  framework:         "pytorch"
  created_at:        "2024-03-15T10:30:00Z"
  created_by:        "training_pipeline_daily"
  training_data:     "interactions_2024-02-14_to_2024-03-14"

  # Quality gates (must pass before deployment)
  offline_metrics:
    click_auc:       0.7823
    watch_time_mae:  12.4
    like_auc:        0.7156

  # Comparison to current production model
  vs_production:
    click_auc_delta: +0.003
    watch_time_delta: -0.2

  # Deployment status
  status:            "canary"     # pending → canary → ramping → production → archived
  traffic_pct:       5
  serving_config:
    batch_size:      64
    max_latency_ms:  50
    gpu_type:        "A100"
```

```python
# Model Registry API
class ModelRegistry:
    def register(self, model_name, version, artifact_path, metrics, metadata):
        """Register a new model version after training."""
        pass

    def promote(self, model_name, version, target_stage):
        """Move model through stages: canary → ramping → production."""
        pass

    def get_production_model(self, model_name):
        """Return the current production model info."""
        pass

    def get_model_history(self, model_name, limit=10):
        """Return recent versions for comparison."""
        pass

    def rollback(self, model_name, target_version):
        """Revert to a previous version."""
        pass
```

**Tools**: MLflow Model Registry, Vertex AI Model Registry, SageMaker Model Registry, or custom-built.

### 4.2 Traffic Router

Decides which model version handles each request. The most critical control plane component for experimentation.

```
Traffic routing strategies:

1. SIMPLE SPLIT (A/B test)
   ┌────────────────┐
   │ Incoming traffic│
   └───────┬────────┘
           │
     hash(user_id) % 100
           │
    ┌──────┴──────┐
    │ < 50        │ ≥ 50
    ▼             ▼
  Model A       Model B
  (control)     (treatment)

2. CANARY (gradual rollout)
   New model gets 1% → 5% → 25% → 50% → 100%
   At each step, check health metrics before proceeding.

   ┌────────────────────────────────────────────────┐
   │ Day 1     Day 2     Day 3     Day 5     Day 7  │
   │ ████░░░░  ████░░░░  ██████░░  ████████  ██████████│
   │  1%        5%       25%       50%       100%   │
   │ canary    monitor   ramp      ramp      full   │
   └────────────────────────────────────────────────┘

3. SHADOW (dark launch)
   Both models run on every request.
   Only Model A's predictions are served to users.
   Model B's predictions are logged for offline comparison.

   Request → Model A (production) → response to user
           → Model B (shadow)     → log only, discard

   Use case: Compare a new model without any user impact.

4. MULTI-ARMED BANDIT (adaptive)
   Start with equal traffic. Automatically shift traffic
   toward the better-performing model.

   Initially:  50% / 50%
   After 1 day: 65% / 35%  (model A winning on watch_time)
   After 3 days: 85% / 15% (model A clearly better)
```

```python
class TrafficRouter:
    """Determine which model version serves a given request."""

    def __init__(self):
        self.experiments = {}  # experiment_id → config

    def get_model_assignment(self, user_id, model_name):
        """
        Returns which model version + config to use for this user.

        The assignment must be DETERMINISTIC for a given user
        (same user always sees the same variant in an A/B test).
        """
        experiment = self.get_active_experiment(model_name)

        if experiment is None:
            return self.get_production_model(model_name)

        # Deterministic hash → consistent user experience
        bucket = hash(f"{user_id}:{experiment.id}") % 10000

        cumulative = 0
        for variant in experiment.variants:
            cumulative += variant.traffic_pct * 100  # e.g., 5% → 500
            if bucket < cumulative:
                return variant.model_version, variant.config

        return experiment.control.model_version, experiment.control.config

    def create_experiment(self, name, model_name, variants):
        """
        Create an A/B test.

        Example:
          create_experiment(
            name="mmoe_v13_test",
            model_name="ranking_mmoe",
            variants=[
              {"name": "control", "version": 12, "traffic_pct": 90},
              {"name": "new_model", "version": 13, "traffic_pct": 5},
              {"name": "new_weights", "version": 12, "traffic_pct": 5,
               "config_override": {"like_weight": 1.0}},  # same model, different weights
            ]
          )
        """
        pass
```

### 4.3 Configuration Manager

Manage runtime parameters that change model behavior **without redeploying the model**.

```
Configurations managed:

1. Score combination weights:
   {
     "click_weight":      0.1,
     "watch_time_weight": 1.0,
     "like_weight":       0.5,
     "finish_weight":     0.3,
     "dislike_weight":    -2.0,
   }
   → Can be changed per A/B test variant, per region, per platform

2. Retrieval parameters:
   {
     "faiss_nprobe":      32,
     "retrieval_k":       300,
     "freshness_boost":   1.2,
     "popularity_source_weight": 0.8,
   }

3. Re-ranking rules:
   {
     "max_same_creator":  3,
     "diversity_lambda":  0.3,
     "exploration_pct":   0.05,
     "clickbait_threshold": 0.7,
   }

4. Feature flags:
   {
     "enable_new_title_embedding": false,
     "enable_session_features":    true,
     "enable_cross_attention":     false,
   }

5. Quality gates:
   {
     "min_click_auc":     0.75,
     "max_p99_latency":   50,
     "max_error_rate":    0.001,
   }
```

```python
class ConfigManager:
    """
    Hierarchical config with overrides.

    Resolution order:
      experiment_override > region_override > model_default > global_default
    """

    def get_config(self, model_name, experiment_id=None, region=None):
        config = self.get_global_defaults()
        config.update(self.get_model_defaults(model_name))

        if region:
            config.update(self.get_region_override(model_name, region))

        if experiment_id:
            config.update(self.get_experiment_override(experiment_id))

        return config
```

### 4.4 Deployment Orchestrator

Manages the actual deployment of model artifacts to serving infrastructure.

```
Deployment workflow:

  1. Training pipeline produces new model → register in Model Registry

  2. Quality gate check (automated):
     ✓ Offline metrics above thresholds?
     ✓ Model artifact valid (loadable, correct input/output shape)?
     ✓ Latency within budget (benchmark on test traffic)?
     ✓ No prediction drift vs. current model (distribution check)?

  3. Canary deployment:
     → Deploy to 1-2 serving pods
     → Route 1-5% of traffic
     → Monitor for 1-24 hours

  4. Automated health check:
     ✓ Error rate < 0.1%?
     ✓ p99 latency < 50ms?
     ✓ Online engagement metrics not degraded?

  5. Gradual ramp:
     5% → 25% → 50% → 100%
     Each step has a "bake time" (wait period) and health check

  6. Full rollout:
     → Old model pods are drained and recycled
     → Old model archived in registry
```

```
Deployment strategies:

  BLUE-GREEN:
  ┌──────────────────────────────────────────────┐
  │ Before:  All traffic → Blue (v12)            │
  │ Deploy:  Load v13 on Green pods (no traffic) │
  │ Switch:  Route all traffic → Green (v13)     │
  │ Verify:  Monitor for 15 min                  │
  │ Done:    Drain Blue, or rollback if issues    │
  └──────────────────────────────────────────────┘
  Pros: Instant rollback (just switch back to Blue)
  Cons: Need 2× infrastructure during transition

  ROLLING UPDATE:
  ┌──────────────────────────────────────────────┐
  │ 10 pods running v12                          │
  │ Update pod 1 → v13, wait, check health       │
  │ Update pod 2 → v13, wait, check health       │
  │ ...                                          │
  │ Update pod 10 → v13                          │
  └──────────────────────────────────────────────┘
  Pros: No extra infrastructure
  Cons: Slower, mixed versions during rollout

  CANARY (most common for ML models):
  ┌──────────────────────────────────────────────┐
  │ 10 pods running v12                          │
  │ Add 1 pod with v13, route 5% traffic to it   │
  │ Monitor online metrics for 24 hours           │
  │ If healthy: gradually shift more traffic       │
  │ If unhealthy: remove canary pod, keep v12     │
  └──────────────────────────────────────────────┘
  Pros: Minimal risk, real traffic validation
  Cons: Needs routing logic, slower rollout
```

### 4.5 Health Monitor

Continuously watches serving metrics and triggers alerts or automatic actions.

```
Metrics monitored:

  Infrastructure metrics:
    ├── p50/p99 latency per model
    ├── Error rate (5xx, timeouts)
    ├── GPU utilization per pod
    ├── Request throughput (QPS)
    └── Queue depth

  Model quality metrics:
    ├── Prediction score distribution (mean, std, percentiles)
    ├── Per-task prediction distribution (P(click), E[watch_time], etc.)
    ├── Feature missing rate
    └── Input feature distribution drift

  Business metrics (lagging, from analytics):
    ├── CTR by model version
    ├── Average watch time by model version
    ├── Like/dislike rate by model version
    └── DAU / retention (long-term)
```

```python
class HealthMonitor:
    """Monitor serving health and trigger automatic actions."""

    def check_health(self, model_name, model_version):
        metrics = self.get_recent_metrics(model_name, model_version, window="5m")

        issues = []

        # Latency check
        if metrics.p99_latency_ms > 50:
            issues.append(f"p99 latency {metrics.p99_latency_ms}ms > 50ms threshold")

        # Error rate check
        if metrics.error_rate > 0.001:
            issues.append(f"Error rate {metrics.error_rate:.4f} > 0.1% threshold")

        # Prediction drift check
        if abs(metrics.avg_click_score - self.baseline.avg_click_score) > 0.05:
            issues.append(f"Click score drift: {metrics.avg_click_score:.3f} "
                         f"vs baseline {self.baseline.avg_click_score:.3f}")

        # NaN/Inf check
        if metrics.nan_prediction_rate > 0:
            issues.append(f"NaN predictions detected: {metrics.nan_prediction_rate:.4f}")

        if issues:
            self.trigger_alert(model_name, model_version, issues)

            if metrics.error_rate > 0.01 or metrics.nan_prediction_rate > 0:
                self.trigger_auto_rollback(model_name)

        return len(issues) == 0

    def trigger_auto_rollback(self, model_name):
        """Automatically revert to last known good version."""
        last_good = self.registry.get_last_production_version(model_name)
        self.orchestrator.rollback(model_name, last_good)
        self.alert(f"AUTO-ROLLBACK: {model_name} reverted to v{last_good}")
```

---

## 5. Traffic Routing Deep Dive: How Percentage-Based Routing Actually Works

This is the core question: when we say "5% canary, 95% production," **what actually happens** at the request level?

### 5.1 The Fundamental Mechanism: Hash Bucketing

Every request carries a `user_id`. We hash it into a fixed number of **buckets** (typically 10,000 or 100), and a **routing table** maps bucket ranges to model backends.

```
Step 1: Hash the user into a bucket
─────────────────────────────────────
  bucket = hash(user_id) % 10000     # 0-9999

  user "alice"   → hash → 4,271  → bucket 4271
  user "bob"     → hash → 812    → bucket 812
  user "charlie" → hash → 9,503  → bucket 9503

  Key property: DETERMINISTIC
  "alice" always gets bucket 4271, every request, every day.
  This ensures she always sees the same model variant (no flickering).

Step 2: Routing table maps buckets → model backend
─────────────────────────────────────────────────────
  Bucket Range     Model Backend       Traffic %
  ────────────     ─────────────       ─────────
  0000 - 0099      model_v13 (canary)     1%
  0100 - 9999      model_v12 (prod)      99%

  "alice"   → bucket 4271 → model_v12 (production)
  "bob"     → bucket 812  → model_v12 (production)
  "charlie" → bucket 9503 → model_v12 (production)
  "dave"    → bucket 42   → model_v13 (canary!)
```

### 5.2 Changing Percentages = Updating the Routing Table

To ramp from 1% → 5% → 30% → 100%, you **only change the routing table**. No model redeployment needed.

```
Day 1: Canary at 1%
  ┌──────────────────────────────────────────────────────────┐
  │ 0    100                                            10000│
  │ ├─────┤──────────────────────────────────────────────────┤│
  │ │ v13 │                    v12                           ││
  │ │ 1%  │                    99%                           ││
  └──────────────────────────────────────────────────────────┘
  routing_table = {(0, 99): "v13", (100, 9999): "v12"}

Day 2: Ramp to 5%
  ┌──────────────────────────────────────────────────────────┐
  │ 0    500                                            10000│
  │ ├─────┤──────────────────────────────────────────────────┤│
  │ │ v13 │                    v12                           ││
  │ │ 5%  │                    95%                           ││
  └──────────────────────────────────────────────────────────┘
  routing_table = {(0, 499): "v13", (500, 9999): "v12"}

Day 4: Ramp to 30%
  ┌──────────────────────────────────────────────────────────┐
  │ 0         3000                                      10000│
  │ ├──────────┤─────────────────────────────────────────────┤│
  │ │   v13    │                    v12                      ││
  │ │   30%    │                    70%                      ││
  └──────────────────────────────────────────────────────────┘
  routing_table = {(0, 2999): "v13", (3000, 9999): "v12"}

Day 6: Full rollout 100%
  ┌──────────────────────────────────────────────────────────┐
  │ 0                                                   10000│
  │ ├────────────────────────────────────────────────────────┤│
  │ │                       v13                              ││
  │ │                      100%                              ││
  └──────────────────────────────────────────────────────────┘
  routing_table = {(0, 9999): "v13"}

Notice: users in buckets 0-99 were ALWAYS on v13 from day 1.
  This is important — the same users stay in the same group.
  We expand the range, never reshuffle.
```

### 5.3 Multi-Experiment Routing (A/B/C Split)

When testing multiple models simultaneously, the routing table has multiple ranges:

```
Experiment: "ranking_model_comparison_q1"

  Bucket Range     Model Backend          Traffic %   Purpose
  ────────────     ─────────────          ─────────   ────────
  0000 - 0499      ranking_ple_v1            5%       Treatment A (PLE)
  0500 - 0999      ranking_mmoe_v14          5%       Treatment B (new MMoE)
  1000 - 1499      ranking_mmoe_v12          5%       Treatment C (old + new weights)
  1500 - 9999      ranking_mmoe_v12         85%       Control

  Config override for Treatment C (same model, different config):
    treatment_c_config = {"like_weight": 1.0, "diversity_lambda": 0.5}
```

### 5.4 Where Does Routing Happen? (Three Layers)

```
                        ┌────────────────────────┐
                        │     User Request        │
                        │   (user_id = "alice")   │
                        └───────────┬────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            ▼                       ▼                       ▼
   Option A: Application      Option B: API Gateway    Option C: Service Mesh
   ─────────────────────      ─────────────────────    ────────────────────────

   Recommendation service     NGINX / Kong / Envoy     Istio / Linkerd
   reads routing table        applies routing rules    applies traffic policy
   and calls the right        at the gateway level     at the network level
   model endpoint                                      (sidecar proxy)

   ┌─────────────────┐       ┌─────────────────┐      ┌─────────────────┐
   │ Rec Service code │       │ API Gateway      │      │ Istio VirtualSvc│
   │                  │       │                  │      │                  │
   │ bucket = hash(id)│       │ route:           │      │ route:           │
   │ if bucket < 500: │       │   weight: 5      │      │   - dest: v13   │
   │   call(model_v13)│       │   backend: v13   │      │     weight: 5   │
   │ else:            │       │   weight: 95     │      │   - dest: v12   │
   │   call(model_v12)│       │   backend: v12   │      │     weight: 95  │
   └─────────────────┘       └─────────────────┘      └─────────────────┘
```

**Which option do companies use?**

```
Approach          Pros                          Cons                    Used By
───────────       ─────                         ─────                   ───────
App-level         Full control, custom logic,   Routing logic in app    YouTube, Netflix
routing           experiment-aware bucketing     code, every service
                                                 needs routing library

API Gateway       Centralized, language-         Limited to weight-     Smaller companies,
routing           agnostic, simple config        based splits, no       API-first
                                                 user-level bucketing   architectures

Service Mesh      Network-level, no code         Weight-based only      Kubernetes-native
(Istio/Envoy)     changes, works across          (random, not user-     companies
                  services                        deterministic!)
```

**Important distinction**: Service mesh traffic splitting (Istio weights) is **random per-request**, not per-user deterministic. A user could see model A on one request and model B on the next. This is fine for load balancing but **bad for A/B testing** (breaks consistency). For proper A/B tests, you need **application-level hash-based routing**.

### 5.5 Complete Implementation

```python
import hashlib
from dataclasses import dataclass

@dataclass
class ModelBackend:
    name: str           # "ranking_mmoe_v13"
    endpoint: str       # "http://model-v13.internal:8080/predict"
    config: dict        # runtime config overrides

@dataclass
class BucketRange:
    start: int          # inclusive
    end: int            # inclusive
    backend: ModelBackend

class TrafficRouter:
    """
    Production traffic router with hash-based deterministic bucketing.

    Usage:
        router = TrafficRouter(num_buckets=10000)
        router.set_routing_table("ranking", [
            BucketRange(0, 499, ModelBackend("v13", "http://v13:8080/predict", {})),
            BucketRange(500, 9999, ModelBackend("v12", "http://v12:8080/predict", {})),
        ])

        backend = router.route("ranking", user_id="alice")
        # → deterministically returns v12 or v13 based on alice's hash
    """

    def __init__(self, num_buckets=10000):
        self.num_buckets = num_buckets
        self.routing_tables = {}  # model_name → List[BucketRange]

    def _hash_to_bucket(self, user_id: str, experiment_salt: str = "") -> int:
        """
        Deterministic hash: same user_id always → same bucket.

        Salt ensures different experiments get independent bucketing.
        (A user in bucket 42 for experiment A is NOT necessarily
         in bucket 42 for experiment B.)
        """
        key = f"{user_id}:{experiment_salt}"
        hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return hash_val % self.num_buckets

    def set_routing_table(self, model_name: str, ranges: list[BucketRange]):
        """
        Atomically update the routing table.
        Called by the control plane when ramping traffic.
        """
        # Validate: ranges must cover all buckets with no gaps/overlaps
        ranges_sorted = sorted(ranges, key=lambda r: r.start)
        assert ranges_sorted[0].start == 0
        assert ranges_sorted[-1].end == self.num_buckets - 1
        for i in range(len(ranges_sorted) - 1):
            assert ranges_sorted[i].end + 1 == ranges_sorted[i + 1].start

        self.routing_tables[model_name] = ranges_sorted

    def route(self, model_name: str, user_id: str) -> ModelBackend:
        """Route a user to the correct model backend."""
        table = self.routing_tables[model_name]
        bucket = self._hash_to_bucket(user_id, experiment_salt=model_name)

        for range_ in table:
            if range_.start <= bucket <= range_.end:
                return range_.backend

        # Fallback: should never reach here if table is valid
        return table[-1].backend

    def get_traffic_summary(self, model_name: str) -> dict:
        """Return current traffic split for debugging/monitoring."""
        table = self.routing_tables[model_name]
        summary = {}
        for range_ in table:
            count = range_.end - range_.start + 1
            pct = count / self.num_buckets * 100
            summary[range_.backend.name] = f"{pct:.1f}%"
        return summary


# --- Example: Ramp a canary from 1% → 5% → 30% → 100% ---

router = TrafficRouter(num_buckets=10000)

v12 = ModelBackend("mmoe_v12", "http://ranking-v12:8080/predict", {})
v13 = ModelBackend("mmoe_v13", "http://ranking-v13:8080/predict", {})

# Day 1: 1% canary
router.set_routing_table("ranking", [
    BucketRange(0, 99, v13),       # 1%
    BucketRange(100, 9999, v12),   # 99%
])
print(router.get_traffic_summary("ranking"))
# → {"mmoe_v13": "1.0%", "mmoe_v12": "99.0%"}

# Day 2: Metrics look good → ramp to 5%
router.set_routing_table("ranking", [
    BucketRange(0, 499, v13),      # 5%
    BucketRange(500, 9999, v12),   # 95%
])

# Day 4: Still healthy → ramp to 30%
router.set_routing_table("ranking", [
    BucketRange(0, 2999, v13),     # 30%
    BucketRange(3000, 9999, v12),  # 70%
])

# Day 6: Full rollout
router.set_routing_table("ranking", [
    BucketRange(0, 9999, v13),     # 100%
])
```

### 5.6 Infrastructure: How Backends Actually Run

The routing table points to model endpoints. Those endpoints are backed by **pods/containers**:

```
Routing table says: "5% → v13 endpoint, 95% → v12 endpoint"

But how many GPU pods back each endpoint?

  Model v12 (95% traffic):
    ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
    │ Pod 1  │ │ Pod 2  │ │ Pod 3  │ │ Pod 4  │ │ Pod 5  │
    │ v12    │ │ v12    │ │ v12    │ │ v12    │ │ v12    │
    │ A100   │ │ A100   │ │ A100   │ │ A100   │ │ A100   │
    └────────┘ └────────┘ └────────┘ └────────┘ └────────┘
    Load balancer distributes across 5 pods (round-robin)

  Model v13 (5% traffic):
    ┌────────┐
    │ Pod 6  │
    │ v13    │
    │ A100   │
    └────────┘
    1 pod is enough for 5% traffic

  As we ramp v13 from 5% → 30% → 100%:
    → Auto-scaler adds more v13 pods
    → Auto-scaler drains v12 pods

  Kubernetes HPA (Horizontal Pod Autoscaler):
    Scale based on GPU utilization or QPS per pod.
    v13 pods: target 70% GPU util → add pod when exceeded
    v12 pods: scale down as traffic decreases
```

```
Kubernetes setup (simplified):

  # Two Deployments, one per model version
  apiVersion: apps/v1
  kind: Deployment
  metadata:
    name: ranking-v12
  spec:
    replicas: 5          # auto-scaled by HPA
    template:
      containers:
        - name: model
          image: ranking-model:v12
          resources:
            limits:
              nvidia.com/gpu: 1
  ---
  apiVersion: apps/v1
  kind: Deployment
  metadata:
    name: ranking-v13
  spec:
    replicas: 1          # starts small for canary
    template:
      containers:
        - name: model
          image: ranking-model:v13
          resources:
            limits:
              nvidia.com/gpu: 1
  ---
  # Two Services (internal endpoints)
  kind: Service
  metadata:
    name: ranking-v12-svc    # → http://ranking-v12-svc:8080
  spec:
    selector:
      app: ranking-v12
  ---
  kind: Service
  metadata:
    name: ranking-v13-svc    # → http://ranking-v13-svc:8080
  spec:
    selector:
      app: ranking-v13

  # Application-level router calls the right service
  # based on user_id hash bucketing
```

### 5.7 Edge Cases

```
Edge Case                    Solution
──────────                   ─────────
Logged-out user (no ID)      Use session_id or device_id for bucketing.
                             If none available, always route to production (safe default).

New user (first request)     Bucket by user_id as normal. New users are uniformly
                             distributed across buckets by the hash function.

User creates new account     Different user_id → different bucket → might switch
                             experiment group. This is acceptable; the new account
                             has no history anyway.

Bot / crawler traffic        Exclude from experiments via user-agent detection.
                             Always route to production to avoid polluting metrics.

Experiment interaction       Two experiments running simultaneously?
                             Use different salts so bucketing is independent.
                             User in treatment for Exp A can be in control for Exp B.
                             Avoid overlapping experiments that modify the same thing.
```

### 5.8 Summary: The Complete Request Path

```
User opens app → API request with user_id
    │
    ▼
[1] Application Router
    bucket = hash("user_123:ranking_exp_42") % 10000 = 327
    routing_table lookup: bucket 327 → v13 (canary range 0-499)
    │
    ▼
[2] HTTP call to model v13 service endpoint
    POST http://ranking-v13-svc:8080/predict
    body: {user_features: ..., candidate_features: [...]}
    │
    ▼
[3] Kubernetes Service → load balances across v13 pods
    (round-robin to whichever v13 pod is healthy)
    │
    ▼
[4] GPU pod runs forward pass → returns scores
    │
    ▼
[5] Response + log: {user_id, model_version, bucket, scores, latency}
    (logged for A/B test analysis and health monitoring)
```

---

## 6. Use Cases

### Use Case 1: Daily Model Refresh

```
Scenario: Training pipeline produces a new ranking model every day.

Flow:
  00:00  Training pipeline starts on yesterday's data
  04:00  Training complete → model artifact saved to S3
  04:05  Register in Model Registry with offline metrics
  04:06  Quality gate: compare offline metrics vs. production
         ✓ click_auc: 0.7823 vs 0.7801 (+0.003) → PASS
         ✓ watch_time_mae: 12.4 vs 12.6 (-0.2) → PASS
  04:10  Deploy to canary (2 pods, 5% traffic)
  04:10-08:00  Monitor canary for 4 hours
         ✓ p99 latency: 42ms → PASS
         ✓ error rate: 0.0002 → PASS
         ✓ online CTR: +0.1% vs control → PASS
  08:00  Ramp to 50%
  12:00  Ramp to 100%
  12:05  Archive previous version

Control plane decisions made:
  1. Should we deploy? (quality gate)
  2. How much traffic? (canary → ramp schedule)
  3. Is it healthy? (monitoring)
  4. When to ramp? (bake time elapsed + metrics pass)
```

### Use Case 2: A/B Testing New Architecture

```
Scenario: Team trained a PLE model to replace MMoE. Need to validate online.

Experiment config:
  {
    "name": "ple_vs_mmoe_q1_2024",
    "model_name": "ranking",
    "duration_days": 14,
    "variants": [
      {"name": "control", "model_version": "mmoe_v12", "traffic_pct": 80},
      {"name": "treatment", "model_version": "ple_v1", "traffic_pct": 20},
    ],
    "primary_metric": "total_watch_time_per_user_per_day",
    "guardrail_metrics": ["dislike_rate", "dau_retention_7d"],
    "min_detectable_effect": 0.5%,  // MDE for statistical significance
  }

Control plane actions:
  Day 0:   Create experiment, deploy PLE model, route 20% traffic
  Day 1-14: Monitor metrics, check guardrails daily
  Day 14:  Statistical significance reached
           Treatment: +1.2% watch time, -0.05% dislike rate → SHIP IT
  Day 15:  Ramp PLE to 100%, archive MMoE
```

### Use Case 3: Emergency Rollback

```
Scenario: New model deployed, suddenly showing high error rate.

Timeline:
  10:00  New model v15 deployed to 100%
  10:15  Health monitor detects error rate spike: 0.5% → 2.3%
  10:15  Alert fired to on-call engineer
  10:16  Auto-rollback triggered (error rate > 1% threshold)
  10:17  v14 re-deployed to all pods
  10:18  Error rate drops to 0.1% (normal)
  10:20  Post-mortem: v15 had a feature schema mismatch
         (training used feature v2, serving still had feature v1)

Control plane actions:
  1. Detected anomaly (health monitor)
  2. Triggered auto-rollback (deployment orchestrator)
  3. Restored previous version (model registry → data plane)
  Total incident time: 3 minutes (mostly automated)
```

### Use Case 4: Configuration-Only Change

```
Scenario: Business wants to boost new creator content. No model change needed.

Before:
  score = 0.1×P(click) + 1.0×E[watch_time] + 0.5×P(like) - 2.0×P(dislike)

After (config change only):
  score = 0.1×P(click) + 1.0×E[watch_time] + 0.5×P(like) - 2.0×P(dislike)
          + 0.3 × new_creator_boost  // added via config, not model code

Control plane actions:
  1. Create A/B experiment with config override
  2. Treatment group gets new_creator_boost = 0.3
  3. No model redeployment needed
  4. Measure impact on new creator video views + overall engagement
  5. If positive → update default config for all traffic
```

### Use Case 5: Multi-Region Deployment

```
Scenario: Model needs different configs per region.

  US region:
    model_version: v13
    like_weight: 0.5
    gpu_pods: 20

  EU region:
    model_version: v13
    like_weight: 0.5
    gpu_pods: 15
    config_override: {diversity_lambda: 0.5}  // EU regulations require more diversity

  APAC region:
    model_version: v12  // still on old version (staggered rollout)
    like_weight: 0.5
    gpu_pods: 25

Control plane manages:
  - Region-aware deployments
  - Config overrides per region
  - Staggered rollouts (deploy to US first, then EU, then APAC)
  - Region-specific health monitoring
```

---

## 7. Netflix's Inference Platform

Netflix has published extensively about their ML infrastructure. Their system is one of the most mature inference control planes in the industry.

### 7.1 Metaflow + Model Lifecycle

Netflix built **Metaflow** (open-sourced) for ML workflow management, and their internal inference platform on top of it.

```
Netflix's model lifecycle:

  Metaflow (training pipeline)
       │
       ▼
  Model Registry (internal, similar to MLflow)
       │
       ├── Model validation (offline metrics, schema check)
       │
       ▼
  Deployment system
       │
       ├── Canary deployment (small % of traffic)
       ├── A/B test framework
       │
       ▼
  Serving infrastructure (Titus — Netflix's container platform)
       │
       ├── Auto-scaling based on traffic patterns
       ├── Regional failover
       │
       ▼
  Monitoring (Atlas — Netflix's metrics system)
       │
       ├── Prediction quality dashboards
       ├── Latency / error monitoring
       └── Auto-rollback on anomaly
```

### 7.2 Netflix's A/B Testing Platform

Netflix runs hundreds of simultaneous A/B tests across recommendations, UI, and streaming quality. Their control plane handles:

```
1. INTERLEAVING (Netflix's preferred method over simple A/B):
   Instead of splitting users into groups, both models rank for the SAME user,
   and results are interleaved into one list.

   Model A ranks: [V1, V2, V3, V4, V5]
   Model B ranks: [V3, V5, V1, V6, V2]
   Interleaved:   [V1, V3, V2, V5, V4, V6]
   (alternate picks from each model's ranking)

   Measure: which model's picks get more plays?

   Advantage: Same user sees both models → much higher statistical power
   (need fewer users to reach significance)

2. ARTWORK PERSONALIZATION A/B:
   Same movie, different thumbnails for different users.
   Control plane manages which thumbnail variant each user sees.
   Each variant is a "model" in the control plane's view.

3. ROW-LEVEL EXPERIMENTS:
   Different ranking models for different rows on the Netflix homepage.
   "Because you watched X" row uses model A.
   "Trending" row uses model B.
   Control plane routes per-row, not just per-user.
```

### 7.3 Netflix's Contextual Bandits for Exploration

Netflix uses the control plane to manage exploration-exploitation trade-offs:

```
Traditional A/B test:
  50% control, 50% treatment — fixed allocation for 2 weeks.
  Problem: If treatment is clearly worse after day 2,
           we're still wasting 50% of traffic on it for 12 more days.

Netflix's approach — Thompson Sampling:
  Day 1: 50% / 50% (start equal)
  Day 2: Model A winning → 55% / 45%
  Day 3: Model A still winning → 65% / 35%
  Day 7: Model A clearly better → 85% / 15%

  The control plane DYNAMICALLY adjusts allocation.
  Users get better recommendations sooner.
  Still collects enough data from Model B to be statistically valid.
```

### 7.4 Netflix's Feature Store Integration

The control plane coordinates with the feature store to ensure feature consistency:

```
Problem: Model v13 was trained with feature_schema_v2.
         Model v12 (still serving 95% traffic) uses feature_schema_v1.
         Both need to run simultaneously during canary.

Netflix's solution:
  Control plane → tells feature store which schema each model needs
  Feature store → serves the right features per model version

  Model Registry entry:
    model_version: 13
    feature_schema: "v2"
    required_features: ["user_id", "title_emb_v2", "session_features", ...]

  When canary routes a request to v13:
    Feature store looks up schema "v2" → returns correct feature set
  When request routes to v12:
    Feature store looks up schema "v1" → returns correct feature set
```

### 7.5 Netflix's Graceful Degradation

```
Netflix's fallback chain (managed by control plane):

  Level 0 (normal):     Full personalized ranking model (GPU, <50ms)
       │ fails
       ▼
  Level 1 (degraded):   Simplified model (CPU, pre-scored cache, <10ms)
       │ fails
       ▼
  Level 2 (basic):      Pre-computed popularity ranking (Redis, <5ms)
       │ fails
       ▼
  Level 3 (emergency):  Static editorial picks (hardcoded list)

  Control plane monitors health and automatically moves down the chain.
  When health recovers, automatically moves back up.

  Circuit breaker pattern:
    If model endpoint fails 5 times in 10 seconds:
      → Open circuit → route to Level 1
      → After 30 seconds, try Level 0 again (half-open)
      → If success → close circuit → back to Level 0
```

---

## 8. Design Patterns

### Pattern 1: Model-as-a-Service

```
Each model type is a separate microservice with its own control plane:

  Retrieval Service          Ranking Service          Re-ranking Service
  ┌─────────────────┐       ┌─────────────────┐      ┌─────────────────┐
  │ Control: v5/v6  │       │ Control: v12/v13│      │ Control: v3     │
  │ Two-Tower model │       │ MMoE model      │      │ Rules + model   │
  │ + FAISS index   │       │ GPU serving     │      │ CPU serving     │
  └────────┬────────┘       └────────┬────────┘      └────────┬────────┘
           │                         │                          │
           └─────────────┬───────────┘──────────────────────────┘
                         │
                Recommendation Orchestrator
                (calls services in sequence)
```

Each service has independent versioning, deployment, and A/B testing. A change to the ranking model doesn't require redeploying retrieval.

### Pattern 2: Feature Flag-Driven Development

```
Instead of deploying new models for every experiment,
use feature flags to toggle behavior within the same model:

  if config.get("enable_cross_attention"):
      user_repr = cross_attention(user_history, candidate)
  else:
      user_repr = average_pool(user_history)

  if config.get("use_watch_time_v2"):
      watch_pred = model.watch_time_tower_v2(features)
  else:
      watch_pred = model.watch_time_tower(features)

Pros: Fast iteration, no deployment needed
Cons: Code complexity grows, dead code accumulates
```

### Pattern 3: Shadow Mode for Validation

```
New model runs in shadow mode before receiving any live traffic:

  Request → Production Model → response to user (as usual)
          → Shadow Model    → logged but NOT served

  After 1 week of shadow logging:
    Compare: shadow_predictions vs production_predictions
    Check: latency, error rate, score distribution
    Check: if shadow model had been serving, would engagement metrics differ?

  Only promote from shadow → canary if shadow results look promising.

  Shadow mode costs compute (running 2 models) but eliminates risk.
```

---

## 9. System Requirements Checklist

| Requirement | Description | Why It Matters |
|---|---|---|
| **Deterministic routing** | Same user always sees same model variant | Consistent user experience, valid A/B test results |
| **Instant rollback** | Revert to previous model in < 5 minutes | Minimize impact of bad deployments |
| **Zero-downtime deployment** | No request failures during model swap | User experience, revenue impact |
| **Config hot-reload** | Change weights/thresholds without restart | Fast iteration, no downtime |
| **Multi-model serving** | Run 2+ model versions simultaneously | A/B testing, canary |
| **Audit trail** | Log who deployed what, when, why | Debugging, compliance |
| **Auto-scaling** | Scale GPU pods based on traffic | Cost efficiency, handle traffic spikes |
| **Feature schema versioning** | Different models can use different feature schemas | Safe model upgrades |
| **Metric collection** | Per-model-version latency, error, quality metrics | Automated health checks |
| **Auto-rollback** | Automatically revert on anomaly detection | Reduce incident response time |
| **Regional awareness** | Deploy to specific regions, stagger rollouts | Blast radius containment |
| **Experiment isolation** | Multiple simultaneous A/B tests don't interfere | Valid experimental results |

---

## 10. Technology Stack

| Component | Open Source Options | Managed Options |
|---|---|---|
| Model Registry | MLflow, DVC, Weights & Biases | Vertex AI, SageMaker |
| Traffic Routing | Istio, Envoy, custom | LaunchDarkly (feature flags) |
| Container Orchestration | Kubernetes + KServe | EKS, GKE |
| Model Serving | TorchServe, Triton Inference Server | Vertex AI Endpoints, SageMaker Endpoints |
| Monitoring | Prometheus + Grafana, Datadog | CloudWatch, Vertex AI Monitoring |
| A/B Testing | Statsig, GrowthBook, custom | Optimizely |
| Feature Store | Feast, Tecton | Vertex AI Feature Store, SageMaker |
| Workflow/Pipeline | Metaflow, Airflow, Kubeflow | Vertex AI Pipelines |

---

## 11. Interview Talking Points

### "Design a control plane for model inference."

> I'd start with four core components: (1) a model registry that tracks versions, artifacts, and offline metrics; (2) a traffic router that deterministically assigns users to model versions for A/B testing and canary deployments; (3) a config manager for runtime parameters like score weights that can change without redeployment; and (4) a health monitor that tracks latency, error rates, and prediction distribution, with auto-rollback triggers.
>
> The deployment flow is: training pipeline registers a new model → quality gates check offline metrics → canary deployment to 5% traffic → automated health monitoring for 4-24 hours → gradual ramp to 100% if healthy, auto-rollback if not.

### "How do you safely deploy a new model?"

> Canary deployment. Deploy the new model to a small number of serving pods and route 1-5% of traffic to it. Monitor for at least 4 hours: check p99 latency (< 50ms), error rate (< 0.1%), and online engagement metrics vs. the control group. If all checks pass, gradually ramp to 25% → 50% → 100% with bake periods between each step. If any metric degrades, auto-rollback to the previous version. The entire flow can be automated with manual override for high-risk changes.

### "How does Netflix handle model experiments?"

> Netflix uses interleaving instead of simple A/B splits for ranking experiments — both models rank for the same user, and results are interleaved into one list. This gives much higher statistical power because each user acts as their own control. They also use Thompson Sampling to dynamically adjust traffic allocation during experiments, so a clearly losing variant gets less traffic over time. Their infrastructure (Metaflow for training, Titus for serving, Atlas for monitoring) supports hundreds of simultaneous experiments.

### "What's the hardest part of a model inference control plane?"

> Feature schema versioning across model versions. When you're running model v12 and v13 simultaneously (during canary), they might expect different feature schemas. The control plane must coordinate with the feature store to serve the correct features per model version. A schema mismatch is the most common cause of silent model degradation — the model runs without errors but predictions are garbage because it received wrong features. Netflix and YouTube both solved this by logging the exact features used at serving time and binding feature schema versions to model versions in the registry.

### "How do you decide between A/B test vs. canary vs. shadow?"

> Shadow mode for high-risk changes (new architecture, major feature changes) — zero user impact, pure validation. Canary for incremental improvements (daily retrained model, minor feature additions) — small blast radius, fast feedback. Full A/B test for measuring business impact of deliberate changes (new score weights, new task) — statistically rigorous comparison. In practice, a new model goes through all three: shadow first (1 week) → canary (1 day) → A/B test (2 weeks) → full rollout.
