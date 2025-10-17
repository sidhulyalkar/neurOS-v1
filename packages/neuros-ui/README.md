# neurOS UI

User interfaces for neurOS - Streamlit dashboard, FastAPI server, and visualizations.

## Features

- **Real-Time Dashboard**: Streamlit-based monitoring interface
- **REST API**: FastAPI server for programmatic access
- **Interactive Visualizations**: Plotly, Matplotlib, Seaborn
- **Live Data Streaming**: WebSocket support for real-time updates
- **Experiment Tracking**: Built-in visualization for training runs

## Installation

```bash
# Minimal installation (visualization only)
pip install neuros-ui

# With Streamlit dashboard
pip install neuros-ui[dashboard]

# With FastAPI server
pip install neuros-ui[api]

# Everything
pip install neuros-ui[all]
```

## Quick Start

### Streamlit Dashboard

```bash
# Launch dashboard
streamlit run -m neuros.ui.dashboard
```

### FastAPI Server

```python
from neuros.ui.api import create_app

# Create API server
app = create_app()

# Run server
# uvicorn neuros.ui.api:app --reload
```

### Visualization

```python
from neuros.viz import plot_eeg_signal, plot_confusion_matrix

# Plot EEG signal
plot_eeg_signal(data, sampling_rate=250)

# Plot model evaluation
plot_confusion_matrix(y_true, y_pred, labels=['Class A', 'Class B'])
```

## Features

### Dashboard
- Live signal monitoring
- Model performance metrics
- Training progress visualization
- Multi-modal data display

### API Endpoints
- `/predict`: Real-time inference
- `/train`: Model training
- `/metrics`: Performance metrics
- `/status`: System health

## Documentation

Full documentation: https://neuros.readthedocs.io

## License

MIT License
