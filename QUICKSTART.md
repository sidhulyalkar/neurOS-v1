# NeurOS Quickstart Guide

Get up and running with NeurOS in **5 minutes**! This guide will walk you through installation, running your first pipeline, and exploring key features.

## Prerequisites

- **Python 3.10+** installed
- **pip** package manager
- **15 MB** of disk space

---

## Installation

### Step 1: Clone or Download

```bash
git clone https://github.com/yourusername/neuros-v1.git
cd neuros-v1
```

Or download and extract the ZIP file.

### Step 2: Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install NeurOS

```bash
pip install -r requirements.txt
pip install -e .
```

### Step 4: Verify Installation

```bash
neuros --help
```

You should see the NeurOS CLI help menu.

---

## Your First Pipeline (60 seconds)

### Run a Simple Pipeline

This runs a mock BCI pipeline for 3 seconds and displays metrics:

```bash
neuros run --duration 3
```

**Output:**
```json
{
  "duration": 3.0,
  "samples": 687,
  "throughput": 229.0,
  "mean_latency": 0.0019,
  "quality_mean": 0.01,
  "quality_std": 0.68,
  "model": "SimpleClassifier",
  "driver": "MockDriver"
}
```

**What just happened?**
- NeurOS created a mock EEG device (8 channels @ 250 Hz)
- Streamed synthetic brain signals for 3 seconds
- Applied bandpass filtering and feature extraction
- Classified signals using a simple logistic regression model
- Reported throughput (~229 samples/sec) and latency (~1.9 ms)

### Run a Benchmark

Test NeurOS performance on your machine:

```bash
neuros benchmark --duration 5
```

This measures:
- **Throughput:** Samples processed per second
- **Latency:** Time from data acquisition to prediction
- **Accuracy:** Classification accuracy (on synthetic data)

---

## Quick Examples

### Example 1: Motor Imagery Classification

Simulate a 2-class motor imagery BCI:

```bash
# Run a pipeline configured for motor imagery
neuros run --duration 5
```

The auto-configuration system detects "motor" and sets up:
- Alpha (8-12 Hz) and Beta (12-30 Hz) band extraction
- Appropriate classifier (e.g., EEGNet or Simple Classifier)
- Real-time processing optimized for motor tasks

### Example 2: Train a Model on Data

If you have a CSV file with features (columns) and labels (last column):

```bash
# Create sample training data
cat > sample_data.csv << EOF
feat1,feat2,feat3,feat4,feat5,label
0.5,1.2,-0.3,0.8,1.1,0
-0.2,0.9,1.5,-0.5,0.3,1
1.0,-0.8,0.2,1.3,-0.6,0
EOF

# Train a model
neuros train --csv sample_data.csv

# Output: Model trained and saved to sample_data.model.pkl
```

### Example 3: Run the Dashboard

Launch the interactive Streamlit dashboard:

```bash
# Install Streamlit first
pip install streamlit

# Launch dashboard
neuros dashboard
```

The dashboard opens in your browser at `http://localhost:8501` and provides:
- Real-time signal visualization
- Pipeline configuration
- Model training interface
- Performance metrics

### Example 4: Start the API Server

Run NeurOS as a REST API service:

```bash
neuros serve --host 127.0.0.1 --port 8000
```

Access the API at `http://127.0.0.1:8000`:
- **Docs:** `http://127.0.0.1:8000/docs` (Swagger UI)
- **Health check:** `GET /health`
- **Train model:** `POST /train`
- **Predict:** `POST /predict`
- **WebSocket stream:** `ws://127.0.0.1:8000/ws/stream`

---

## Understanding the Pipeline

### What is a Pipeline?

A NeurOS pipeline consists of:

```
[Driver] → [Processing Agent] → [Feature Extraction] → [Model] → [Predictions]
```

1. **Driver:** Acquires data from hardware or simulation
2. **Processing Agent:** Applies filters and preprocessing
3. **Feature Extraction:** Computes band power, entropy, etc.
4. **Model:** Classifies or predicts based on features
5. **Output:** Returns predictions with timestamps and metrics

### Available Drivers

| Driver | Description | Status |
|--------|-------------|--------|
| `MockDriver` | Simulated EEG device (default) | ✅ Built-in |
| `BrainFlowDriver` | Support for OpenBCI, Muse, Emotiv, etc. | ⚠️ Requires `brainflow` |
| `DatasetDriver` | Load from NumPy/CSV files | ✅ Built-in |
| `VideoDriver` | Webcam/video file input | ✅ Built-in |
| `MotionSensorDriver` | Accelerometer, gyroscope | ✅ Built-in |
| `ECGDriver` | Electrocardiogram | ✅ Built-in |
| `EMGDriver` | Electromyography (muscle signals) | ✅ Built-in |

### Available Models

| Model | Description | Best For |
|-------|-------------|----------|
| `SimpleClassifier` | Logistic regression (default) | Quick prototyping |
| `RandomForestModel` | Ensemble tree-based model | Robust classification |
| `SVMModel` | Support vector machine | Small datasets |
| `EEGNetModel` | Specialized CNN for EEG | Motor imagery, P300 |
| `CNNModel` | 1D convolutional network | Time-series data |
| `TransformerModel` | Attention-based model | Long sequences |
| `GBDTModel` | Gradient boosting | Tabular features |

---

## Using Real Hardware

### OpenBCI Cyton

Install BrainFlow:

```bash
pip install brainflow
```

Create a Python script:

```python
# openbci_demo.py
import asyncio
from neuros.pipeline import Pipeline
from neuros.drivers import BrainFlowDriver
from neuros.models import EEGNetModel

async def main():
    # OpenBCI Cyton board (board_id=0)
    driver = BrainFlowDriver(
        board_id=0,
        sampling_rate=250.0,
        channels=8,
        serial_port="/dev/ttyUSB0"  # Adjust for your system
    )

    model = EEGNetModel(n_channels=8, n_classes=2)

    pipeline = Pipeline(
        driver=driver,
        model=model,
        fs=250.0,
    )

    # Run for 10 seconds
    metrics = await pipeline.run(duration=10.0)
    print(metrics)

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:

```bash
python openbci_demo.py
```

### Emotiv EPOC

```python
driver = BrainFlowDriver(
    board_id=1,  # Emotiv board ID
    sampling_rate=128.0,
    channels=14,
)
```

See [BrainFlow documentation](https://brainflow.readthedocs.io/) for board IDs.

---

## Auto-Configuration Magic ✨

NeurOS can automatically configure pipelines based on task descriptions:

```python
from neuros.autoconfig import generate_pipeline_for_task

# Automatically select driver, model, and frequency bands
pipeline = generate_pipeline_for_task("motor imagery EEG classification")

# Or for SSVEP (steady-state visual evoked potential)
pipeline = generate_pipeline_for_task("SSVEP 4-class visual BCI")

# Or for multi-modal
pipeline = generate_pipeline_for_task("emotion recognition from EEG and facial video")
```

The system detects keywords like:
- **Modalities:** EEG, video, motion, EMG, ECG
- **Tasks:** motor imagery, P300, SSVEP, emotion
- **Models:** Automatically selects appropriate classifier

---

## Multi-Modal Pipelines

NeurOS supports combining multiple data sources:

```python
import asyncio
from neuros.pipeline import MultiModalPipeline
from neuros.drivers import BrainFlowDriver, VideoDriver, MotionSensorDriver
from neuros.models import CompositeModel

async def main():
    # Create drivers for each modality
    eeg_driver = BrainFlowDriver(board_id=0, channels=8)
    video_driver = VideoDriver(source=0)  # Webcam
    motion_driver = MotionSensorDriver(device="/dev/ttyACM0")

    # Composite model fuses features from all modalities
    model = CompositeModel(n_modalities=3, n_classes=3)

    pipeline = MultiModalPipeline(
        drivers=[eeg_driver, video_driver, motion_driver],
        model=model,
    )

    metrics = await pipeline.run(duration=5.0)
    print(metrics)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Next Steps

### Learn More

- **[README.md](README.md)** - Project overview and features
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute
- **[AUDIT.md](AUDIT.md)** - Current project status and roadmap
- **[docs/](docs/)** - Technical documentation and white papers

### Tutorials

1. **Motor Imagery Classification** - Train a BCI to detect imagined hand movements
2. **P300 Speller** - Build a brain-controlled keyboard
3. **SSVEP BCI** - Frequency-based visual interface
4. **Multi-Modal Emotion Recognition** - Combine EEG, facial video, and physiological signals

(Coming soon in `notebooks/` directory)

### Get Help

- **GitHub Issues:** Report bugs or request features
- **Discussions:** Ask questions and share ideas
- **Documentation:** Check the `docs/` folder

---

## Common Issues

### Issue: `ModuleNotFoundError: No module named 'brainflow'`

**Solution:** BrainFlow is optional. Install with:
```bash
pip install brainflow
```

Or use the mock driver for testing.

### Issue: `RuntimeError: LSL binary library file was not found`

**Solution:** Lab Streaming Layer is optional. Install with:
```bash
# macOS
brew install labstreaminglayer/tap/lsl

# Linux
conda install -c conda-forge liblsl

# Or skip LSL features
```

### Issue: Dashboard not launching

**Solution:** Install Streamlit:
```bash
pip install streamlit
```

### Issue: Tests failing

**Solution:** Install test dependencies:
```bash
pip install pytest pytest-asyncio httpx nbformat
```

---

## Performance Tips

### Optimize Latency

1. **Use GPU acceleration** (if available):
   ```bash
   pip install torch  # Already in requirements
   ```

2. **Reduce processing overhead:**
   ```python
   pipeline = Pipeline(
       driver=driver,
       model=model,
       adaptation=False,  # Disable adaptive thresholding
       filters=[],  # Minimal filtering
   )
   ```

3. **Use appropriate window sizes:**
   - Smaller windows = lower latency, less context
   - Larger windows = higher latency, better features

### Scale for High Throughput

1. **Use the Constellation pipeline** for production:
   ```bash
   neuros constellation --duration 60 --output-dir /data/experiment1
   ```

2. **Enable Kafka streaming** for distributed processing

3. **Deploy with Docker** for consistent performance

---

## What's Next?

You've learned the basics! Now you can:

- ✅ Run pipelines with mock data
- ✅ Train and save models
- ✅ Use the dashboard and API
- ✅ Understand the pipeline architecture

**Advanced topics:**
- Custom driver development
- Deep learning models (EEGNet, Transformers)
- Real-time feedback and closed-loop BCIs
- Cloud deployment (AWS, GCP, Azure)
- Multi-user collaboration

Check out the full documentation in the `docs/` folder and join our community!

---

**Welcome to NeurOS - Let's build the future of brain-computer interfaces! 🧠🚀**
