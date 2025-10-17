# neurOS Architecture

Complete architectural overview of the neurOS BCI platform.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        neurOS Platform                       │
│                                                              │
│  ┌────────────┐  ┌──────────────┐  ┌──────────┐  ┌────────┐│
│  │  Drivers   │→ │  Processing  │→ │  Models  │→ │ Export ││
│  │            │  │              │  │          │  │        ││
│  │  • Mock    │  │  • Filters   │  │  • Simple│  │  • NWB ││
│  │  • Dataset │  │  • Features  │  │  • SVM   │  │  • PKL ││
│  │  • BrainFlow│  │  • CSP       │  │  • RF    │  │  • NPZ ││
│  │  • LSL     │  │  • Riemannian│  │  • KNN   │  │        ││
│  └────────────┘  └──────────────┘  └──────────┘  └────────┘│
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Orchestration & Agents                   │   │
│  │  • Pipeline Management  • Real-time Scheduling        │   │
│  │  • Multi-modal Fusion  • Adaptive Processing         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Module Organization

### Monorepo Structure

```
neuros-v1/
├── neuros/                      # Core package
│   ├── drivers/                 # Data acquisition
│   ├── processing/              # Signal processing
│   ├── models/                  # Classification/regression
│   ├── agents/                  # Orchestration
│   ├── sync/                    # Synchronization
│   ├── cloud/                   # Cloud integrations
│   ├── io/                      # I/O utilities
│   └── ui/                      # User interfaces
│
├── packages/                    # Modular sub-packages
│   ├── neuros-core/            # Core functionality
│   ├── neuros-drivers/         # Driver implementations
│   ├── neuros-processing/      # Signal processing
│   ├── neuros-models/          # ML models
│   ├── neuros-cloud/           # Cloud services
│   ├── neuros-ui/              # UI components
│   └── neuros-foundation/      # Foundation models
│
├── tests/                       # Test suite
│   ├── integration/            # Integration tests
│   └── test_*.py               # Unit tests
│
├── notebooks/                   # Tutorials
├── docs/                        # Documentation
├── scripts/                     # Utilities
└── demo/                        # Demos & examples
```

## Core Components

### 1. Drivers (Data Acquisition)

```
┌─────────────────────────────────────┐
│         BaseDriver Interface         │
├─────────────────────────────────────┤
│  + get_data() → np.ndarray          │
│  + get_duration() → float            │
│  + channels: int                     │
│  + sampling_rate: float              │
└─────────────────────────────────────┘
             ▲
             │
     ┌───────┴───────┬──────────────┬──────────────┐
     │               │              │              │
┌────────┐    ┌─────────────┐ ┌───────────┐ ┌──────────┐
│  Mock  │    │   Dataset   │ │ BrainFlow │ │   LSL    │
│ Driver │    │   Driver    │ │  Driver   │ │  Driver  │
└────────┘    └─────────────┘ └───────────┘ └──────────┘
```

**Responsibilities:**
- Abstract hardware differences
- Provide consistent data format
- Handle sampling rate conversions
- Manage connection lifecycle

**Data Flow:**
```
Hardware/File → Driver.get_data() → (n_samples, n_channels) → Processing
```

### 2. Signal Processing

```
┌─────────────────────────────────────────────────────┐
│              Processing Pipeline                     │
│                                                      │
│  Raw Data → [Filters] → [Feature Extraction] → Features
│                │              │                           │
│            ┌───┴──┐       ┌───┴───┐                    │
│            │      │       │       │                    │
│       Bandpass Smooth   BandPow  CSP                  │
│       Filter   Filter   Extract                       │
└─────────────────────────────────────────────────────┘
```

#### Filtering Layer

```python
# Filter Chain
data → BandpassFilter → SmoothingFilter → NotchFilter → filtered_data
         (0.02ms)          (0.01ms)        (0.02ms)
```

**Optimization:**
- SOS (Second-Order Sections) for stability
- Streaming filters maintain state
- Vectorized multi-channel operations

#### Feature Extraction Layer

```python
# Feature Extraction
filtered_data → BandPowerExtractor → features
                    (0.4-3ms)
                       ↓
                  [delta, theta, alpha, beta, gamma]
                  per channel
```

**Advanced Features:**
```
┌──────────────────────────────────────┐
│     Advanced Feature Extractors       │
├──────────────────────────────────────┤
│  • CommonSpatialPatterns (CSP)       │
│  • RiemannianFeatures                │
│  • WaveletFeatures                   │
│  • TimeFrequencyFeatures             │
└──────────────────────────────────────┘
```

### 3. Models

```
┌────────────────────────────────────┐
│       BaseModel Interface           │
├────────────────────────────────────┤
│  + train(X, y) → None              │
│  + predict(X) → np.ndarray         │
│  + predict_proba(X) → np.ndarray   │
│  + save(path) → None               │
│  + load(path) → BaseModel          │
└────────────────────────────────────┘
         ▲
         │
    ┌────┴───────┬────────────┬──────────┐
    │            │            │          │
┌────────┐  ┌────────┐  ┌──────────┐  ┌──────┐
│ Simple │  │  SVM   │  │  Random  │  │ KNN  │
│Classif.│  │ Model  │  │  Forest  │  │Model │
└────────┘  └────────┘  └──────────┘  └──────┘
```

**Performance Characteristics:**
```
Model            Latency    Accuracy   Use Case
SimpleClassifier 0.03ms     75-80%     Real-time control
SVMModel         0.08ms     80-85%     General purpose
RandomForest     0.8ms      85-90%     High accuracy needed
KNN              0.05ms     72-82%     Few-shot learning
```

### 4. Pipeline Orchestration

```
┌────────────────────────────────────────────────────────┐
│                    Pipeline                             │
├────────────────────────────────────────────────────────┤
│  Configuration:                                         │
│    • driver: BaseDriver                                │
│    • model: BaseModel                                  │
│    • filters: List[Filter]                             │
│    • bands: Dict[str, Tuple[float, float]]            │
│                                                         │
│  Execution Flow:                                        │
│    1. driver.get_data() → raw_data                    │
│    2. Apply filters → filtered_data                   │
│    3. Extract features → feature_vector               │
│    4. model.predict() → prediction                    │
│    5. Apply adaptation → final_output                 │
│                                                         │
│  Metrics:                                               │
│    • samples, throughput, latency                      │
│    • quality metrics (via QualityMonitor)             │
└────────────────────────────────────────────────────────┘
```

**Orchestrator Agent:**
```
┌──────────────────────────────────┐
│      Orchestrator Agent           │
│                                   │
│  Manages:                         │
│  • ProcessingAgent (filters)     │
│  • FeatureAgent (extraction)     │
│  • ModelAgent (prediction)       │
│  • AdaptationAgent (threshold)   │
│                                   │
│  Coordination:                    │
│  • Async task scheduling         │
│  • Data flow management          │
│  • Error handling                │
│  • Metric collection             │
└──────────────────────────────────┘
```

## Data Flow Architecture

### Single-Modal Pipeline

```
┌────────┐     ┌──────────┐     ┌──────────┐     ┌───────┐     ┌────────┐
│ Driver │────▶│ Filters  │────▶│ Features │────▶│ Model │────▶│ Output │
│        │     │ (0.03ms) │     │ (0.4ms)  │     │(0.08ms│     │        │
└────────┘     └──────────┘     └──────────┘     └───────┘     └────────┘
  250 Hz      (250 samples)    (n_features)      (classes)      Action
   EEG         4-30 Hz band      Band power         SVM          Command
```

**Latency Budget (4 channels, 250 Hz, 1 sec window):**
```
Component         Time (ms)    % of Total
─────────────────────────────────────────
Data acquisition    4.0          62%
Filtering           0.03          0.5%
Feature extraction  0.4           6%
Model prediction    0.08          1%
Adaptation          0.02          0.3%
Output              2.0          31%
─────────────────────────────────────────
Total              ~6.5ms        100%
```

### Multi-Modal Pipeline

```
┌─────────┐     ┌──────────┐     ┌──────────┐
│ EEG     │────▶│ Process1 │────▶│Features1 │──┐
│ Driver  │     │          │     │          │  │
└─────────┘     └──────────┘     └──────────┘  │
                                                 │
┌─────────┐     ┌──────────┐     ┌──────────┐  │   ┌────────┐    ┌────────┐
│ GSR     │────▶│ Process2 │────▶│Features2 │──┼──▶│ Fusion │───▶│ Model  │
│ Driver  │     │          │     │          │  │   │        │    │        │
└─────────┘     └──────────┘     └──────────┘  │   └────────┘    └────────┘
                                                 │
┌─────────┐     ┌──────────┐     ┌──────────┐  │
│ PPG     │────▶│ Process3 │────▶│Features3 │──┘
│ Driver  │     │          │     │          │
└─────────┘     └──────────┘     └──────────┘
```

**Feature Fusion Strategies:**
1. **Early fusion:** Concatenate raw signals
2. **Feature-level fusion:** Concatenate extracted features (default)
3. **Decision-level fusion:** Ensemble of model predictions

## Real-Time Processing Architecture

### Streaming Pipeline

```
┌───────────────────────────────────────────────────────┐
│           Real-Time Streaming Pipeline                 │
│                                                        │
│  Data Stream ──┬──▶ RingBuffer ──▶ StreamingFilter ──┐│
│                │       (O(1))          (Stateful)     ││
│                │                                       ││
│                └──▶ Historical Data                   ││
│                                                        ││
│         ┌──────────────────────────────────────┘      │
│         │                                              │
│         ▼                                              │
│  StreamingFeatureExtractor ──▶ Model ──▶ Output       │
│         (Windowed)            (Cached)                 │
└───────────────────────────────────────────────────────┘
```

**Key Components:**

1. **RingBuffer:** O(1) append and retrieval
   ```python
   buffer = RingBuffer(capacity=1000, n_channels=8)
   buffer.append(new_data)  # O(1)
   recent = buffer.get_last(250)  # O(1)
   ```

2. **StreamingFilter:** Maintains state between calls
   ```python
   filt = StreamingFilter(b, a, n_channels=8)
   filtered = filt.process(new_sample)  # No warm-up
   ```

3. **Windowed Processing:** Extract features when window is ready
   ```python
   extractor = StreamingFeatureExtractor(
       window_size=250,
       hop_size=50,
       feature_func=band_power
   )
   features = extractor.process(new_data)  # None until ready
   ```

### Low-Latency Architecture

```
┌─────────────────────────────────────────────────────┐
│         LowLatencyPipeline Architecture              │
│                                                      │
│  Input Sample                                        │
│     │                                                │
│     ▼                                                │
│  ┌──────────────┐                                   │
│  │  RingBuffer  │  ← Preallocated, fixed size       │
│  └──────┬───────┘                                   │
│         │                                            │
│         ▼                                            │
│  ┌──────────────┐                                   │
│  │  Filters     │  ← Streaming, stateful            │
│  └──────┬───────┘                                   │
│         │                                            │
│         ▼                                            │
│  ┌──────────────┐                                   │
│  │  Features    │  ← Incremental extraction         │
│  └──────┬───────┘                                   │
│         │                                            │
│         ▼                                            │
│  ┌──────────────┐                                   │
│  │  Model       │  ← Fast inference (float32)       │
│  └──────┬───────┘                                   │
│         │                                            │
│         ▼                                            │
│  Output (< 5ms typical)                             │
└─────────────────────────────────────────────────────┘
```

## Advanced Features Architecture

### Common Spatial Patterns (CSP)

```
Training Phase:
┌──────────────────────────────────────────────────────┐
│  X_train (trials × channels × samples)               │
│     │                                                 │
│     ├─▶ Class 0 trials ──▶ Covariance Matrix C₀     │
│     └─▶ Class 1 trials ──▶ Covariance Matrix C₁     │
│                                │                      │
│                                ▼                      │
│            Solve: C₀·w = λ·(C₀ + C₁)·w              │
│                                │                      │
│                                ▼                      │
│               Select top-k eigenvalues                │
│                                │                      │
│                                ▼                      │
│                    Spatial Filters W                  │
└──────────────────────────────────────────────────────┘

Inference Phase:
┌──────────────────────────────────────────────────────┐
│  X_test (trials × channels × samples)                │
│                │                                      │
│                ▼                                      │
│        X_filtered = W · X_test                       │
│                │                                      │
│                ▼                                      │
│   Features = log(var(X_filtered, axis=time))        │
└──────────────────────────────────────────────────────┘
```

### Riemannian Geometry

```
Feature Space:
┌────────────────────────────────────────────────┐
│  EEG Trial → Covariance Matrix                 │
│             ↓                                   │
│        SPD Manifold                            │
│             ↓                                   │
│    Riemannian Distance to Class Means          │
│             ↓                                   │
│         Features                               │
└────────────────────────────────────────────────┘

Distance Metrics:
• Riemannian: d(A,B) = ||log(A^(-1/2)·B·A^(-1/2))||
• Log-Euclidean: d(A,B) = ||log(A) - log(B)||
• Euclidean: d(A,B) = ||A - B||
```

## Optimization Architecture

### Hierarchical Optimization

```
Level 1: Algorithmic
├─ Use CSP/Riemannian instead of raw features
├─ Reduce frequency bands from 7 to 3-4
└─ Select optimal model complexity

Level 2: Implementation
├─ Use optimized modules (OptimizedBandPowerExtractor)
├─ Enable caching for repeated computations
├─ Use SOS filters instead of IIR form
└─ Process in float32 instead of float64

Level 3: Architecture
├─ Use streaming components (RingBuffer)
├─ Enable batch processing
├─ Preallocate buffers
└─ Minimize memory allocation

Level 4: System
├─ Use parallel processing (n_jobs=-1)
├─ Enable SIMD/vectorization
├─ Pin to performance cores
└─ Disable unnecessary services
```

### Memory Architecture

```
┌──────────────────────────────────────────────┐
│          Memory Management                    │
│                                               │
│  Preallocated Buffers (Fixed Size):          │
│  ┌─────────────────────────────────┐         │
│  │  RingBuffer   [1000 × 8 × f32] │         │
│  │  Filter State [max_order × 8]   │         │
│  │  Feature Vec  [n_features × f32] │         │
│  └─────────────────────────────────┘         │
│                                               │
│  Dynamic Allocation (Minimized):             │
│  ┌─────────────────────────────────┐         │
│  │  Model temporary buffers         │         │
│  │  Occasional reallocation         │         │
│  └─────────────────────────────────┘         │
└──────────────────────────────────────────────┘
```

## Deployment Architectures

### Local/Edge Deployment

```
┌────────────────────────────────────────┐
│         Local Machine / Edge Device     │
│                                         │
│  ┌──────────┐        ┌───────────┐    │
│  │ Hardware │───────▶│  neurOS   │    │
│  │  (EEG)   │        │  Pipeline │    │
│  └──────────┘        └─────┬─────┘    │
│                             │           │
│                             ▼           │
│                      ┌────────────┐    │
│                      │   Output   │    │
│                      │  (Action)  │    │
│                      └────────────┘    │
└────────────────────────────────────────┘
```

### Cloud-Augmented Deployment

```
┌────────────────────────┐         ┌─────────────────────┐
│      Edge Device        │         │      Cloud          │
│                        │         │                     │
│  ┌──────┐  ┌────────┐ │         │  ┌───────────────┐ │
│  │ EEG  │─▶│ neurOS │─┼────────▶│  │  Advanced     │ │
│  │      │  │ (Fast) │ │  Stream  │  │  Processing   │ │
│  └──────┘  └───┬────┘ │         │  │  (Foundation) │ │
│                │       │         │  └───────┬───────┘ │
│                ▼       │         │          │         │
│            Immediate   │         │          ▼         │
│            Response    │         │     Deep Analysis  │
│                        │         │     + Retraining   │
└────────────────────────┘         └─────────────────────┘
```

## Package Dependencies

```
Core Dependencies:
├─ numpy (arrays, linear algebra)
├─ scipy (signal processing)
└─ scikit-learn (models, metrics)

Optional Dependencies:
├─ torch (deep learning models)
├─ brainflow (hardware drivers)
├─ pylsl (Lab Streaming Layer)
├─ pywavelets (wavelet analysis)
├─ pynwb (NWB format support)
├─ fastapi (REST API)
├─ kafka-python (streaming)
└─ numba (JIT compilation)
```

## Testing Architecture

```
┌────────────────────────────────────────────────┐
│                Test Suite                       │
│                                                 │
│  Unit Tests (230+)                             │
│  ├─ Models                                     │
│  ├─ Processing                                 │
│  ├─ Drivers                                    │
│  └─ Utilities                                  │
│                                                 │
│  Integration Tests (30)                        │
│  ├─ Pipeline workflows                         │
│  ├─ Multi-modal                                │
│  ├─ Export/import                              │
│  └─ Performance                                │
│                                                 │
│  Performance Tests                             │
│  ├─ Latency benchmarks                         │
│  ├─ Throughput tests                           │
│  └─ Memory profiling                           │
└────────────────────────────────────────────────┘
```

---

For more details, see:
- [API Reference](API_REFERENCE.md)
- [Optimization Guide](OPTIMIZATION.md)
- [Contributing Guidelines](../CONTRIBUTING.md)
