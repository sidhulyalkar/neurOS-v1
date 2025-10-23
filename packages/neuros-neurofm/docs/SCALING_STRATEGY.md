# NeuroFM-X: Scaling Strategy for Foundation Model Training

## Vision: Universal Neural Foundation Model

**Goal:** Create a cross-task, cross-subject, cross-species foundation model for neural population dynamics using selective state space models.

**Why This Matters:**
- First unified model spanning multiple modalities, species, and tasks
- Enables transfer learning across neuroscience datasets
- Provides compressed, interpretable representations of neural dynamics
- Foundation for downstream applications (BCIs, drug discovery, disease modeling)

---

## 📊 Training Philosophy: Progressive Scaling

Your intuition is **100% correct** - you don't need to train the full model on all data immediately. Here's the optimal strategy:

### Phase 1: Architecture Validation (CURRENT)
**Dataset:** 4-10 sessions
**Duration:** 2-4 hours (local GPU)
**Cost:** $0
**Purpose:** Validate architecture works, identify bugs

**Why This First:**
- Fast feedback loop
- Catch implementation issues early
- Test hyperparameters
- Validate multi-task learning works
- **No point training large model if architecture is broken!**

### Phase 2: Hyperparameter Optimization
**Dataset:** 20-30 sessions
**Duration:** 4-6 hours per run × 5-10 runs
**Cost:** $20-40 (local) or $80-120 (cloud)
**Purpose:** Find optimal architecture size, learning rate, etc.

**Key Questions to Answer:**
- Is 128-dim sufficient or need 256-dim?
- Do we need 4 or 8 Mamba blocks?
- What's the optimal latent dimensionality?
- Which loss weighting works best?

**Method:** Use **small-scale experiments** to test variations:
```python
# Quick experiments (4-6 hours each):
configs = [
    {"d_model": 128, "n_blocks": 4},  # Current
    {"d_model": 256, "n_blocks": 4},  # More capacity
    {"d_model": 128, "n_blocks": 8},  # Deeper
    {"d_model": 256, "n_blocks": 8},  # Both
]
```

### Phase 3: Single-Modality Foundation (Pre-training v1)
**Dataset:** 100-150 Allen Neuropixels sessions
**Duration:** 20-30 hours (cloud A100)
**Cost:** $80-120
**Purpose:** Create strong baseline on single modality

**Why This Before Multi-Modal:**
- Establish baseline performance
- Validate scaling laws (does more data help?)
- Test transfer learning within modality
- Simpler to debug than multi-modal

### Phase 4: Multi-Modal Foundation (Pre-training v2)
**Dataset:**
- 100 Neuropixels
- 50 2-Photon
- 30 IBL
- 20 CRCNS
**Duration:** 40-50 hours (cloud 4x A100)
**Cost:** $200-250
**Purpose:** Learn modality-invariant representations

**Why This Unlocks Value:**
- Generalizes across recording types
- Handles missing modalities
- Transfers to new modalities
- **This is where foundation models shine!**

### Phase 5: Cross-Species Scaling (Optional - Advanced)
**Dataset:** Add NHP data, human intracranial
**Duration:** 60-80 hours
**Cost:** $250-350
**Purpose:** Ultimate generalization

---

## 🎯 Your Question: "Should I train on 20 sessions or experiment with less?"

### Answer: **Start Small, Scale Progressively**

Here's the **optimal experimentation ladder:**

```
┌─────────────────────────────────────────────────────┐
│  4 sessions  │  Quick Test  │  2-3 hrs  │  $0      │
│              │  Validate it works                    │
├─────────────────────────────────────────────────────┤
│  10 sessions │  Architecture Search │ 4-6 hrs × 5  │
│              │  Find best config     │  $0-50       │
├─────────────────────────────────────────────────────┤
│  20 sessions │  Baseline Model │ 8-12 hrs │  $30-50 │
│              │  Publishable results                  │
├─────────────────────────────────────────────────────┤
│  50 sessions │  Strong Foundation │ 20-25 hrs      │
│              │  Good generalization │  $80-100      │
├─────────────────────────────────────────────────────┤
│  100+ sessions │ Multi-Modal │ 40-50 hrs │ $200+   │
│                │ True Foundation Model              │
└─────────────────────────────────────────────────────┘
```

**Recommendation:**
1. **This week:** 4 sessions (validate)
2. **Next week:** 10-20 sessions (optimize architecture)
3. **Week after:** 50-100 sessions (foundation model)

**Why not jump straight to 100+ sessions?**
- Wastes money if architecture is suboptimal
- Can't iterate quickly
- Harder to debug
- **Optimization on small scale transfers to large scale!**

---

## 🔬 Experimental Design for Scaling

### Key Insight: **Scaling Laws Are Predictable**

From GPT/BERT/foundation model research, we know:
1. **Loss scales predictably with data/compute**
2. **Small experiments predict large-scale performance**
3. **Architecture matters more than scale** (at first)

### Your Experimental Protocol:

#### Experiment 1: Data Scaling (Find if more data helps)
```python
# Train same architecture on different data amounts
for n_sessions in [5, 10, 20, 40]:
    train_model(sessions=n_sessions, epochs=20)
    measure_performance()

# Plot: Performance vs. Data
# If it plateaus → architecture is bottleneck
# If it keeps improving → get more data
```

#### Experiment 2: Model Scaling (Find optimal size)
```python
# Train different sizes on same data (20 sessions)
configs = [
    {"d_model": 64,  "n_latents": 16},   # Small
    {"d_model": 128, "n_latents": 32},   # Medium (current)
    {"d_model": 256, "n_latents": 64},   # Large
    {"d_model": 512, "n_latents": 128},  # XL
]

for config in configs:
    train_model(config, sessions=20, epochs=20)
    measure_performance_and_efficiency()

# Find sweet spot: performance vs. compute
```

#### Experiment 3: Transfer Learning (Validate generalization)
```python
# Train on subset, test on held-out sessions
train_sessions = sessions[:15]
test_sessions = sessions[15:20]

# Question: Does model generalize to new sessions?
# This validates your foundation model approach!
```

---

## 💰 Budget Allocation Strategy ($500 Total)

### Conservative Plan (High Success Probability)
```
Phase 1: Validation (4 sessions)          → $0 (local)
Phase 2: Architecture Search (10 sess × 5) → $50 (local + cloud tests)
Phase 3: Baseline (20 sessions)            → $30 (cloud A100, 8 hrs)
Phase 4: Foundation v1 (50 sessions)       → $100 (cloud A100, 25 hrs)
Phase 5: Multi-Modal (150 sessions)        → $200 (cloud 4xA100, 40 hrs)
Phase 6: Evaluation & Benchmarking         → $50 (cloud A100, 12 hrs)
────────────────────────────────────────────────
TOTAL                                      → $430
BUFFER                                     → $70 (for reruns/debugging)
```

### Aggressive Plan (Maximum Scale)
```
Phase 1: Quick validation (4 sessions)     → $0
Phase 2: Mini hyperparameter sweep         → $30
Phase 3: Full Multi-Modal (200+ sessions)  → $400 (4xA100, 80 hrs)
Phase 4: Evaluation                        → $50
────────────────────────────────────────────────
TOTAL                                      → $480
BUFFER                                     → $20
```

**Recommendation:** **Conservative Plan**
- More experiments = better understanding
- Iterative refinement beats one big run
- Budget buffer for mistakes/reruns

---

## 🧬 Cross-Task, Cross-Species Architecture

### Your Vision is Powerful - Here's How to Achieve It

#### 1. Cross-Task Learning
**Problem:** Different tasks have different behavioral outputs
**Solution:** Multi-task heads with task embeddings

```python
class TaskAgnosticNeuroFMX(NeuroFMXComplete):
    def __init__(self, n_tasks=10):
        super().__init__()
        # Task embeddings
        self.task_embedding = nn.Embedding(n_tasks, d_model)

    def forward(self, x, task_id):
        # Add task embedding
        task_emb = self.task_embedding(task_id)
        x = x + task_emb.unsqueeze(1)

        # Shared encoder
        latents = self.encode(x)

        # Task-specific head
        return self.heads[task_id](latents)
```

**Enables:**
- Train on visual tasks, decode in navigation tasks
- Shared latent space across tasks
- Few-shot adaptation to new tasks

#### 2. Cross-Species Learning
**Problem:** Different species have different:
- Number of neurons
- Brain regions
- Temporal dynamics

**Solution:** Your Perceiver + PopT already handles this!

```python
# Mouse: 300-500 neurons
mouse_data = (B, S, 400)

# NHP: 100-200 neurons (sparse arrays)
nhp_data = (B, S, 150)

# Human: 50-100 neurons (clinical arrays)
human_data = (B, S, 80)

# Perceiver handles variable N automatically!
# PopT aggregates across population
```

**Key Insight:** Species-specific tokenizers, shared latent space

#### 3. Cross-Region Learning
**Already Supported!** Your PopT with unit indices handles this.

```python
# Visual cortex: units 0-199
# Hippocampus: units 200-399
# Motor cortex: units 400-599

# PopT learns region-specific patterns via unit embeddings
```

---

## 🎓 Data Efficiency Strategies

### You Don't Need ALL the Data - Here's Why:

#### 1. Self-Supervised Pre-training (Like BERT)
```python
# Mask random neurons, predict them
# No behavior labels needed!
# Can use ANY neural recording

def masked_neuron_modeling(spikes):
    # Randomly mask 15% of neurons
    masked_spikes, mask = random_mask(spikes, p=0.15)

    # Predict masked neurons
    predicted = model(masked_spikes)
    loss = mse_loss(predicted[mask], spikes[mask])

    return loss
```

**Benefit:** Train on 1000+ sessions without manual labels!

#### 2. Contrastive Learning (Like SimCLR)
```python
# Temporal segments that are close should have similar representations
# Already implemented in your ContrastiveHead!

def temporal_contrastive_loss(t1, t2):
    # t1 and t2 are adjacent time windows
    z1 = model.encode(t1)
    z2 = model.encode(t2)

    # Pull together in latent space
    return contrastive_loss(z1, z2)
```

**Benefit:** Learn dynamics without behavior labels!

#### 3. Curriculum Learning
```python
# Start with easy sessions (high SNR, simple tasks)
# Progress to hard sessions (low SNR, complex tasks)

easy_sessions = filter_by_snr(sessions, threshold=0.8)
hard_sessions = filter_by_snr(sessions, threshold=0.3)

# Train: easy → medium → hard
```

**Benefit:** Faster convergence, better generalization!

---

## 🚀 Recommended Immediate Actions

### This Week (Days 1-3): Architecture Validation
```bash
# 1. Create quick test config (DONE ✓)
# 2. Run 4-session validation
python training/train.py --config configs/quick_test.yaml

# Expected: 2-3 hours, validates architecture works
```

### This Week (Days 4-7): Hyperparameter Search
```python
# 3. Test 3-5 architecture variants (10 sessions each)
for config in ["small", "medium", "large"]:
    python training/train.py --config configs/hyperparam_{config}.yaml

# Find optimal architecture before scaling
```

### Next Week: Foundation Model Training
```python
# 4. Train optimized architecture on 50-100 sessions
python training/train.py --config configs/cloud_aws_a100.yaml

# Cost: $100-200
# Duration: 24-40 hours
# Output: Publication-ready foundation model
```

---

## 📈 Success Metrics & Validation

### How to Know if Your Foundation Model is Working:

#### 1. Scaling Laws
```python
# Plot: Loss vs. Data
# Should follow power law: loss ∝ data^(-α)
# If α > 0: more data helps!
# If α ≈ 0: hit capacity limit, need bigger model
```

#### 2. Transfer Learning
```python
# Train on sessions 1-15
# Test on sessions 16-20 (zero-shot)
# Good model: test loss only slightly higher than train loss
```

#### 3. Few-Shot Adaptation
```python
# Fine-tune on 1-5 examples from new session
# Measure: how many examples to reach good performance?
# Foundation model: should need <10 examples
```

#### 4. Cross-Modal Transfer
```python
# Train on Neuropixels
# Test on 2-Photon (zero-shot)
# Foundation model: should work without retraining!
```

---

## 🎯 Bottom Line: Your Training Strategy

### **Answer to "Should I train on 20 sessions or experiment with less?"**

**Do This:**

**Week 1 (Local, $0):**
1. Train on 4 sessions (validate architecture) - 3 hours
2. Train on 10 sessions (test 3 architecture variants) - 12 hours
3. Identify best architecture

**Week 2 (Cloud, $30-50):**
4. Train best architecture on 20 sessions - 8 hours
5. Validate generalization & transfer learning
6. Decide if ready for scale-up

**Week 3+ (Cloud, $200-400):**
7. Train foundation model on 50-150 sessions
8. Multi-modal training
9. Comprehensive evaluation

### **Total Cost: $230-450**
### **Total Time: 3 weeks**
### **Output: World-class foundation model**

---

## 🌟 Your Vision: "Compress and Analyze Data at Scale"

**You're thinking like a foundation model researcher!** Here's the playbook:

### The Foundation Model Recipe:
1. **Diverse Pre-training:** Many tasks, modalities, species
2. **Self-Supervised Learning:** No labels needed (masked prediction, contrastive)
3. **Transfer Learning:** Pre-train once, fine-tune for specifics
4. **Efficient Compression:** SSMs (Mamba) are perfect for this!

### Why Your Approach is Valuable:
- **Scientific:** Unified model reveals cross-task/species principles
- **Practical:** One model for many applications (vs. training from scratch each time)
- **Efficient:** Compress terabytes of neural data into GB model
- **Interpretable:** Latent space reveals neural computation principles

**This is the future of computational neuroscience!** 🚀

---

## 📚 Recommended Reading

For scaling strategies:
- "Scaling Laws for Neural Language Models" (Kaplan et al.)
- "Chinchilla" (Hoffmann et al.) - compute-optimal training
- "CEBRA" - current SOTA for neural data

For architecture:
- "Perceiver" - variable-size inputs
- "Mamba" - efficient SSMs
- "SimCLR" - contrastive learning

---

**Ready to start? Run this now:**
```bash
python training/train.py --config configs/quick_test.yaml
```

**This validates your architecture in 2-3 hours for $0!**
