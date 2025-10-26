# Unified Mechanistic Interpretability Reporting System

## Overview

The Unified Mechanistic Interpretability Reporting System provides a comprehensive framework for generating professional, publication-ready reports from all mechanistic interpretability analyses in NeuroFMX.

## Features

### Core Capabilities

- **Multiple Output Formats**: HTML, Markdown, PDF (via HTML)
- **Rich Content Types**: Text, figures, tables, metrics, code snippets
- **Interactive Visualizations**: Both Matplotlib (static) and Plotly (interactive)
- **Professional Styling**: Responsive HTML with modern CSS
- **Experiment Tracking**: MLflow and Weights & Biases integration
- **Automated Analysis**: Run all mech-int analyses and generate unified reports

### Report Components

1. **Sections**: Markdown-formatted text content
2. **Figures**: Matplotlib or Plotly visualizations (embedded in HTML)
3. **Tables**: Pandas DataFrames with professional formatting
4. **Metrics**: Scalar values with units and descriptions
5. **Code Snippets**: Syntax-highlighted code blocks

## Quick Start

### Basic Report Creation

```python
from neuros_neurofm.interpretability.reporting import MechIntReport

# Create report
report = MechIntReport(
    output_dir="./reports",
    title="My Mechanistic Interpretability Report"
)

# Add content
report.add_section(
    "Introduction",
    "This report analyzes the causal structure of our model."
)

# Add metrics
report.add_metric(
    "Model Accuracy",
    95.5,
    unit="%",
    description="Test set accuracy"
)

# Add figure
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
ax.set_title("Example Plot")
report.add_figure(fig, "Simple quadratic relationship")

# Add table
import pandas as pd
df = pd.DataFrame({
    'Layer': ['L1', 'L2', 'L3'],
    'Parameters': [1024, 2048, 1024]
})
report.add_table(df, "Layer statistics")

# Generate reports
html_path = report.generate_html()
md_path = report.generate_markdown()

print(f"HTML report: {html_path}")
print(f"Markdown report: {md_path}")
```

### Unified Reporter (All Analyses)

```python
from neuros_neurofm.interpretability.reporting import UnifiedMechIntReporter
import torch

# Create model and data
model = YourModel()
data = torch.randn(100, 128)

# Configure analyses
config = {
    'num_samples': 100,
    'sae': {
        'input_dim': 512,
        'hidden_dim': 1024,
        'k_active': 32
    }
}

# Create reporter
reporter = UnifiedMechIntReporter(
    model=model,
    data=data,
    config=config,
    output_dir="./reports"
)

# Run all analyses
results = reporter.run_all_analyses()

# Or run specific analyses
results = reporter.run_selected_analyses([
    "causal_graphs",
    "energy_flow",
    "sae_features"
])

# Generate comprehensive report
report = reporter.generate_report()

# Export to HTML/Markdown
report.generate_html("full_analysis.html")
report.generate_markdown("full_analysis.md")
```

## API Reference

### MechIntReport

Main class for creating reports.

#### Constructor

```python
MechIntReport(output_dir, title="NeuroFMX Mechanistic Interpretability Report")
```

**Parameters:**
- `output_dir` (str or Path): Directory to save reports
- `title` (str): Report title

#### Methods

##### add_section(title, content)

Add a markdown section to the report.

```python
report.add_section(
    "Analysis Overview",
    """
    This section provides an overview of the analysis.

    **Key Points:**
    - Point 1
    - Point 2
    """
)
```

##### add_figure(fig, caption, name=None)

Add a matplotlib or plotly figure.

```python
# Matplotlib
fig, ax = plt.subplots()
ax.plot(x, y)
report.add_figure(fig, "Training loss over time")

# Plotly
import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(x=x, y=y))
report.add_figure(fig, "Interactive plot")
```

##### add_table(df, caption, name=None)

Add a pandas DataFrame as a table.

```python
df = pd.DataFrame({
    'Metric': ['Accuracy', 'Loss'],
    'Value': [0.95, 0.12]
})
report.add_table(df, "Model performance metrics")
```

##### add_metric(name, value, unit="", description="")

Add a scalar metric.

```python
report.add_metric(
    "Test Accuracy",
    94.5,
    unit="%",
    description="Accuracy on held-out test set"
)
```

##### add_code(code, language="python")

Add a code snippet.

```python
code = """
def analyze_model(model, data):
    return model(data)
"""
report.add_code(code, language="python")
```

##### generate_html(filename="report.html")

Generate HTML report with embedded figures.

```python
html_path = report.generate_html("my_report.html")
```

Returns: `Path` to generated HTML file

##### generate_markdown(filename="report.md")

Generate Markdown report.

```python
md_path = report.generate_markdown("my_report.md")
```

Returns: `Path` to generated Markdown file

##### export_to_mlflow(mlflow_client=None, run_id=None)

Log report to MLflow.

```python
import mlflow

mlflow.set_experiment("my_experiment")
with mlflow.start_run():
    report.export_to_mlflow()
```

##### export_to_wandb(wandb_run=None)

Log report to Weights & Biases.

```python
import wandb

wandb.init(project="my_project")
report.export_to_wandb()
wandb.finish()
```

### UnifiedMechIntReporter

Class for running all mechanistic interpretability analyses and generating unified reports.

#### Constructor

```python
UnifiedMechIntReporter(model, data, config=None, output_dir="reports")
```

**Parameters:**
- `model`: Model to analyze
- `data`: Data for analysis
- `config` (dict): Configuration for analyses
- `output_dir` (str or Path): Directory to save reports

#### Methods

##### run_all_analyses()

Run all available analyses.

```python
results = reporter.run_all_analyses()
```

**Analyses run:**
1. Causal graphs
2. Energy flow
3. Topology
4. SAE features
5. Alignment
6. Dynamics
7. Counterfactuals
8. Attribution

Returns: `dict` of analysis results

##### run_selected_analyses(analyses)

Run specific analyses.

```python
results = reporter.run_selected_analyses([
    "causal_graphs",
    "energy_flow"
])
```

Returns: `dict` of analysis results

##### generate_report(title=None)

Generate unified report from all analyses.

```python
report = reporter.generate_report(
    title="Comprehensive Analysis Report"
)
```

Returns: `MechIntReport` object

## HTML Report Structure

The generated HTML reports have the following structure:

```html
<!DOCTYPE html>
<html>
<head>
    <!-- Professional styling with responsive CSS -->
    <!-- Plotly.js for interactive plots -->
    <!-- Highlight.js for code syntax highlighting -->
</head>
<body>
    <div class="container">
        <header>
            <!-- Report title and timestamp -->
        </header>

        <nav>
            <!-- Navigation menu with links to sections -->
        </nav>

        <div class="content">
            <!-- Summary section with key metrics -->

            <!-- Individual analysis sections -->
            <!-- Each section can contain: -->
            <!--   - Text content -->
            <!--   - Figures (embedded or interactive) -->
            <!--   - Tables -->
            <!--   - Code snippets -->
        </div>

        <footer>
            <!-- Attribution and copyright -->
        </footer>
    </div>
</body>
</html>
```

### Key Features

1. **Responsive Design**: Works on desktop, tablet, and mobile
2. **Interactive Navigation**: Smooth scrolling to sections
3. **Embedded Figures**: Base64-encoded images for portability
4. **Interactive Plots**: Plotly charts with zoom, pan, hover
5. **Professional Styling**: Modern gradient headers, shadows, spacing
6. **Metrics Dashboard**: Grid layout for key metrics
7. **Code Highlighting**: Syntax-highlighted code snippets
8. **Print-Friendly**: Optimized CSS for printing/PDF export

## Configuration

### Analysis Configuration

```python
config = {
    # Number of samples to use for analysis
    'num_samples': 100,

    # SAE configuration
    'sae': {
        'input_dim': 512,
        'hidden_dim': 1024,
        'k_active': 32
    },

    # Causal graph settings
    'causal': {
        'significance_level': 0.05,
        'max_lag': 5
    },

    # Energy flow settings
    'energy': {
        'normalize': True,
        'threshold': 0.01
    }
}
```

## Examples

### Example 1: Quick Analysis Report

```python
from neuros_neurofm.interpretability.reporting import MechIntReport

report = MechIntReport("./quick_report")

# Add overview
report.add_section("Overview", "Quick analysis of model behavior")

# Add key findings
report.add_metric("Sparsity", 0.67, description="Feature sparsity")
report.add_metric("Accuracy", 94.2, unit="%")

# Generate
report.generate_html()
```

### Example 2: Comprehensive Analysis

```python
from neuros_neurofm.interpretability.reporting import UnifiedMechIntReporter

# Setup
reporter = UnifiedMechIntReporter(
    model=model,
    data=data,
    config={'num_samples': 200}
)

# Run analyses
results = reporter.run_all_analyses()

# Generate report
report = reporter.generate_report()
report.generate_html("comprehensive_analysis.html")
```

### Example 3: Custom Report with Multiple Figures

```python
import matplotlib.pyplot as plt
import plotly.graph_objects as go

report = MechIntReport("./custom_report")

# Matplotlib figure
fig1, ax = plt.subplots(2, 2, figsize=(12, 10))
# ... create subplots ...
report.add_figure(fig1, "Multi-panel analysis")

# Plotly interactive figure
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=x, y=y, mode='lines+markers'))
fig2.update_layout(title='Interactive Analysis')
report.add_figure(fig2, "Interactive exploration")

# Generate
report.generate_html()
```

### Example 4: MLflow Integration

```python
import mlflow
from neuros_neurofm.interpretability.reporting import MechIntReport

# Create report
report = MechIntReport("./mlflow_reports")
# ... add content ...

# Log to MLflow
mlflow.set_experiment("neurofmx_interpretability")
with mlflow.start_run():
    report.export_to_mlflow()
```

### Example 5: W&B Integration

```python
import wandb
from neuros_neurofm.interpretability.reporting import MechIntReport

# Initialize W&B
wandb.init(project="neurofmx", name="interpretability_run")

# Create report
report = MechIntReport("./wandb_reports")
# ... add content ...

# Log to W&B
report.export_to_wandb()
wandb.finish()
```

## Advanced Usage

### Custom Templates

```python
from neuros_neurofm.interpretability.reporting import ReportTemplate

# Get template for specific analysis
template = ReportTemplate.get_analysis_template("causal_graphs")

# Get custom CSS
css = ReportTemplate.get_custom_css()
```

### Batch Report Generation

```python
# Generate reports for multiple models
models = [model1, model2, model3]
data = get_data()

for i, model in enumerate(models):
    reporter = UnifiedMechIntReporter(
        model=model,
        data=data,
        output_dir=f"./reports/model_{i}"
    )

    results = reporter.run_all_analyses()
    report = reporter.generate_report(title=f"Model {i} Analysis")
    report.generate_html(f"model_{i}_report.html")
```

### Programmatic Report Composition

```python
# Create report programmatically
def create_analysis_report(analysis_results):
    report = MechIntReport("./automated_reports")

    # Add sections based on results
    for analysis_name, result in analysis_results.items():
        report.add_section(
            analysis_name.replace('_', ' ').title(),
            f"Analysis completed with {len(result)} findings."
        )

        # Add metrics
        if 'metrics' in result:
            for metric_name, metric_value in result['metrics'].items():
                report.add_metric(metric_name, metric_value)

        # Add figures
        if 'figures' in result:
            for fig_name, fig in result['figures'].items():
                report.add_figure(fig, f"{fig_name} visualization")

    return report
```

## Best Practices

### 1. Report Organization

- Start with an overview/summary section
- Group related analyses together
- Use clear, descriptive section titles
- Include key metrics at the top

### 2. Visualization Guidelines

- Use interactive Plotly charts for exploration
- Use Matplotlib for publication-quality static figures
- Ensure figures have clear titles and labels
- Include informative captions

### 3. Metric Selection

- Focus on actionable metrics
- Include units and descriptions
- Highlight critical thresholds
- Show both absolute and relative values

### 4. Performance

- Limit figure resolution for large reports (dpi=150 is good)
- Use sampling for large datasets
- Generate markdown for quick preview, HTML for final report

### 5. Integration

- Log to experiment tracking early and often
- Version your reports with git
- Include report generation in CI/CD pipelines

## Troubleshooting

### Issue: Figures not showing in HTML

**Solution**: Ensure figures are generated before calling `add_figure()`. The report embeds base64-encoded images.

### Issue: Plotly figures not interactive

**Solution**: Check that Plotly CDN is accessible. The report uses `include_plotlyjs='cdn'`.

### Issue: Memory issues with large reports

**Solution**:
- Generate markdown instead of HTML
- Reduce figure DPI
- Process data in batches
- Use sampling

### Issue: MLflow/W&B export fails

**Solution**:
- Verify libraries are installed
- Check active run exists
- Ensure network connectivity
- Verify credentials

## Contributing

To extend the reporting system:

1. Add new report sections in `ReportTemplate`
2. Implement custom visualizations in analysis modules
3. Create new export methods for other platforms
4. Enhance HTML/CSS styling

## License

Part of the NeuroFMX mechanistic interpretability suite.
