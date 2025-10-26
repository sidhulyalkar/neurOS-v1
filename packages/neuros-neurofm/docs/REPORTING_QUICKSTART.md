# Reporting System Quick Start Guide

## 5-Minute Quick Start

### 1. Basic Report (2 minutes)

```python
from neuros_neurofm.interpretability.reporting import MechIntReport
import matplotlib.pyplot as plt

# Create report
report = MechIntReport(output_dir="./my_report")

# Add content
report.add_section("Overview", "This is my analysis.")
report.add_metric("Accuracy", 95.5, unit="%")

# Add a plot
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
report.add_figure(fig, "Results")

# Generate
report.generate_html()  # Open the HTML file in your browser!
```

### 2. Comprehensive Analysis (5 minutes)

```python
from neuros_neurofm.interpretability.reporting import UnifiedMechIntReporter
import torch

# Your model and data
model = MyModel()
data = torch.randn(100, 128)

# Create unified reporter
reporter = UnifiedMechIntReporter(
    model=model,
    data=data,
    output_dir="./full_analysis"
)

# Run all analyses and generate report
results = reporter.run_all_analyses()
report = reporter.generate_report()

# Export
report.generate_html("complete_analysis.html")
```

## Common Use Cases

### Use Case 1: Training Report

```python
report = MechIntReport("./training_reports")

# Add training metrics
report.add_metric("Final Loss", 0.023)
report.add_metric("Best Accuracy", 96.8, unit="%")
report.add_metric("Training Time", 4.5, unit="hours")

# Add training curves
fig, ax = plt.subplots()
ax.plot(epochs, losses)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
report.add_figure(fig, "Training loss over time")

report.generate_html("training_report.html")
```

### Use Case 2: Model Comparison

```python
report = MechIntReport("./comparison")

# Compare multiple models
results_df = pd.DataFrame({
    'Model': ['Model A', 'Model B', 'Model C'],
    'Accuracy': [94.5, 96.2, 95.8],
    'Speed': [45, 38, 42],
    'Size': [125, 98, 110]
})

report.add_table(results_df, "Model comparison")
report.generate_html("comparison.html")
```

### Use Case 3: Causal Analysis

```python
from neuros_neurofm.interpretability.causal_graphs import CausalGraphAnalyzer

# Analyze causal structure
analyzer = CausalGraphAnalyzer(model)
graph = analyzer.build_causal_graph(data)

# Create report
report = MechIntReport("./causal_analysis")
report.add_section("Causal Discovery", "Discovered causal relationships")
report.add_metric("Causal Edges", len(graph.edges()))

# Visualize
fig = analyzer.visualize_graph(graph)
report.add_figure(fig, "Causal graph structure")

report.generate_html("causal_report.html")
```

### Use Case 4: MLflow Integration

```python
import mlflow

mlflow.set_experiment("my_experiment")

with mlflow.start_run():
    # Create report
    report = MechIntReport("./mlflow_reports")
    report.add_metric("Accuracy", 95.5)

    # Train model, run analyses...

    # Log everything to MLflow
    report.export_to_mlflow()
```

### Use Case 5: Batch Processing

```python
# Process multiple datasets
datasets = [data1, data2, data3]

for i, data in enumerate(datasets):
    report = MechIntReport(f"./reports/dataset_{i}")

    # Run analysis
    results = analyze(model, data)

    # Add to report
    report.add_metric("Mean", results.mean())
    report.add_metric("Std", results.std())

    report.generate_html(f"dataset_{i}_report.html")
```

## Cheat Sheet

### Import
```python
from neuros_neurofm.interpretability.reporting import MechIntReport, UnifiedMechIntReporter
```

### Create Report
```python
report = MechIntReport(output_dir="./reports", title="My Report")
```

### Add Content
```python
# Section
report.add_section("Title", "Content in **markdown**")

# Metric
report.add_metric("Name", value, unit="units", description="...")

# Figure (matplotlib or plotly)
report.add_figure(fig, "Caption")

# Table (pandas DataFrame)
report.add_table(df, "Caption")

# Code
report.add_code("code here", language="python")
```

### Generate
```python
html_path = report.generate_html("report.html")
md_path = report.generate_markdown("report.md")
```

### Export
```python
# MLflow
report.export_to_mlflow()

# W&B
report.export_to_wandb()
```

## Tips & Tricks

### 1. Interactive Plots

Use Plotly for interactive visualizations:

```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='markers'))
fig.update_layout(title='Interactive Plot')

report.add_figure(fig, "Explore this plot!")
```

### 2. Multiple Subplots

Create comprehensive visualizations:

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(x, y1)
axes[0, 1].plot(x, y2)
axes[1, 0].plot(x, y3)
axes[1, 1].plot(x, y4)

plt.tight_layout()
report.add_figure(fig, "Four-panel analysis")
```

### 3. Rich Markdown

Use markdown formatting in sections:

```python
report.add_section("Results", """
## Key Findings

1. **Accuracy improved** by 5%
2. *Inference time* reduced by 20%
3. Model size decreased to 80MB

**Conclusion:** The new approach is superior.
""")
```

### 4. Organized Metrics

Group related metrics:

```python
# Performance metrics
report.add_metric("Accuracy", 95.5, unit="%", description="Test accuracy")
report.add_metric("F1 Score", 0.94, description="Test F1")
report.add_metric("AUC-ROC", 0.98, description="ROC curve area")

# Resource metrics
report.add_metric("Latency", 23.5, unit="ms", description="Average latency")
report.add_metric("Memory", 512, unit="MB", description="Peak memory")
```

### 5. Custom Styling

The HTML reports use professional styling automatically. No configuration needed!

## Next Steps

- Read the full [Reporting Guide](REPORTING_GUIDE.md)
- Explore [examples](../examples/generate_mechint_reports.py)
- Check out [tests](../tests/test_reporting.py) for more usage patterns
- Integrate with your experiment tracking workflow

## FAQ

**Q: Can I customize the HTML styling?**
A: The HTML template is in the `generate_html()` method. You can modify it or subclass `MechIntReport`.

**Q: How do I add custom analysis types?**
A: Extend `UnifiedMechIntReporter` and add methods following the `_run_*()` pattern.

**Q: Can I export to PDF?**
A: Generate HTML and use your browser's "Print to PDF" feature for professional PDFs.

**Q: How do I add multiple figures to one section?**
A: Call `add_figure()` multiple times. Each creates a new section automatically.

**Q: What if my analysis fails?**
A: `UnifiedMechIntReporter` catches exceptions and includes error info in the report.

## Examples in the Wild

Check these example scripts:

1. `examples/generate_mechint_reports.py` - Comprehensive examples
2. `tests/test_reporting.py` - Unit tests showing usage
3. `docs/REPORTING_GUIDE.md` - Full documentation

Happy reporting!
