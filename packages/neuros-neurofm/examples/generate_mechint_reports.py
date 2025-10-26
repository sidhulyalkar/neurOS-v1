"""
Comprehensive examples for the Unified Mechanistic Interpretability Reporting System.

This script demonstrates various use cases for generating professional reports
from mechanistic interpretability analyses.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
import torch
import torch.nn as nn

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuros_neurofm.interpretability.reporting import (
    MechIntReport,
    UnifiedMechIntReporter,
    ReportTemplate
)


def example_1_basic_report():
    """Example 1: Creating a basic mechanistic interpretability report."""
    print("\n" + "="*80)
    print("Example 1: Basic Report Creation")
    print("="*80)

    # Create report
    report = MechIntReport(
        output_dir="./reports/example1",
        title="Basic Mechanistic Interpretability Report"
    )

    # Add introduction
    report.add_section(
        "Introduction",
        """
This report demonstrates the basic capabilities of the unified reporting system.
It includes various types of content: text sections, figures, tables, metrics, and code.

The reporting system supports:
- Multiple output formats (HTML, Markdown)
- Interactive visualizations (Plotly)
- Static plots (Matplotlib)
- Data tables (Pandas)
- Code snippets with syntax highlighting
- Integration with MLflow and W&B
        """
    )

    # Add metrics
    report.add_metric("Model Accuracy", 94.5, unit="%", description="Test set accuracy")
    report.add_metric("Training Time", 3.5, unit="hours", description="Total training duration")
    report.add_metric("Parameters", 125000000, description="Total model parameters")
    report.add_metric("Active Features", 1024, description="Number of active SAE features")

    # Add matplotlib figure
    fig1, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Training curves
    epochs = np.arange(1, 51)
    train_loss = 2.0 * np.exp(-epochs/10) + 0.1
    val_loss = 2.0 * np.exp(-epochs/10) + 0.2

    axes[0, 0].plot(epochs, train_loss, label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_loss, label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Feature activation distribution
    activations = np.random.lognormal(0, 1, 10000)
    axes[0, 1].hist(activations, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Activation Magnitude')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Feature Activation Distribution')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Layer-wise sparsity
    layers = np.arange(1, 13)
    sparsity = 0.3 + 0.4 * np.sin(layers / 2) + np.random.normal(0, 0.05, len(layers))
    sparsity = np.clip(sparsity, 0, 1)

    axes[1, 0].bar(layers, sparsity, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Sparsity')
    axes[1, 0].set_title('Layer-wise Feature Sparsity')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Plot 4: Attention pattern
    attention = np.random.rand(8, 8)
    attention = attention / attention.sum(axis=1, keepdims=True)

    im = axes[1, 1].imshow(attention, cmap='viridis', aspect='auto')
    axes[1, 1].set_xlabel('Key Position')
    axes[1, 1].set_ylabel('Query Position')
    axes[1, 1].set_title('Attention Pattern Heatmap')
    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    report.add_figure(fig1, "Comprehensive model analysis showing training dynamics, feature distributions, sparsity patterns, and attention mechanisms")

    # Add Plotly interactive figure
    fig2 = go.Figure()

    # Energy flow through layers
    layers_list = list(range(12))
    energy = 100 * np.exp(-np.array(layers_list) / 5)

    fig2.add_trace(go.Scatter(
        x=layers_list,
        y=energy,
        mode='lines+markers',
        name='Energy Flow',
        line=dict(color='royalblue', width=3),
        marker=dict(size=10)
    ))

    fig2.update_layout(
        title='Energy Flow Through Model Layers',
        xaxis_title='Layer',
        yaxis_title='Energy (arbitrary units)',
        hovermode='x unified',
        template='plotly_white'
    )

    report.add_figure(fig2, "Interactive visualization of information energy flow through model layers")

    # Add table
    layer_stats = pd.DataFrame({
        'Layer': [f'Layer {i}' for i in range(1, 13)],
        'Parameters': [1024*512, 512*512, 512*512, 512*512, 512*1024, 1024*1024,
                      1024*1024, 1024*512, 512*512, 512*512, 512*256, 256*100],
        'Sparsity': [f"{s:.2%}" for s in sparsity],
        'Activation': ['ReLU'] * 11 + ['Softmax']
    })

    report.add_table(layer_stats, "Detailed layer-by-layer statistics including parameters, sparsity, and activation functions")

    # Add code snippet
    code = """
# Example: Computing causal effects
from neuros_neurofm.interpretability.causal_graphs import CausalGraphAnalyzer

analyzer = CausalGraphAnalyzer(model)
graph = analyzer.build_causal_graph(data)
effects = analyzer.compute_causal_effects(graph, data)

# Visualize top causal connections
analyzer.visualize_graph(graph, top_k=20)
    """

    report.add_code(code, language="python")

    # Generate reports
    html_path = report.generate_html("basic_report.html")
    md_path = report.generate_markdown("basic_report.md")

    print(f"\n✓ Generated HTML report: {html_path}")
    print(f"✓ Generated Markdown report: {md_path}")


def example_2_causal_analysis_report():
    """Example 2: Causal graph analysis report."""
    print("\n" + "="*80)
    print("Example 2: Causal Graph Analysis Report")
    print("="*80)

    report = MechIntReport(
        output_dir="./reports/example2",
        title="Causal Graph Analysis Report"
    )

    # Add overview
    report.add_section(
        "Causal Structure Discovery",
        """
This report presents the causal structure discovered in the neural network.
We analyze how different components of the model causally influence each other
during computation.

**Key Findings:**
- Identified 48 causal relationships between layers
- Found 3 critical bottleneck nodes
- Discovered 2 distinct information pathways
- Measured average causal effect strength of 0.67
        """
    )

    # Add metrics
    report.add_metric("Causal Nodes", 24, description="Total nodes in causal graph")
    report.add_metric("Causal Edges", 48, description="Total causal relationships")
    report.add_metric("Graph Density", 0.17, description="Edge density of causal graph")
    report.add_metric("Avg Effect Strength", 0.67, description="Mean causal effect magnitude")

    # Simulate causal graph structure
    import networkx as nx

    # Create a sample causal graph
    G = nx.DiGraph()
    num_layers = 6
    nodes_per_layer = 4

    # Add nodes
    for layer in range(num_layers):
        for node in range(nodes_per_layer):
            G.add_node(f"L{layer}N{node}", layer=layer)

    # Add edges (connections between adjacent layers)
    for layer in range(num_layers - 1):
        for src in range(nodes_per_layer):
            # Each node connects to 2-3 nodes in next layer
            num_connections = np.random.randint(2, 4)
            targets = np.random.choice(nodes_per_layer, num_connections, replace=False)
            for tgt in targets:
                weight = np.random.uniform(0.3, 1.0)
                G.add_edge(f"L{layer}N{src}", f"L{layer+1}N{tgt}", weight=weight)

    # Visualize causal graph
    fig, ax = plt.subplots(figsize=(14, 10))

    # Position nodes in layers
    pos = {}
    for layer in range(num_layers):
        for node in range(nodes_per_layer):
            pos[f"L{layer}N{node}"] = (layer, nodes_per_layer - node)

    # Draw graph
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    nx.draw_networkx_edges(
        G, pos,
        edge_color=weights,
        edge_cmap=plt.cm.viridis,
        width=2,
        arrows=True,
        arrowsize=20,
        ax=ax,
        edge_vmin=0,
        edge_vmax=1
    )

    ax.set_title("Causal Graph Structure", fontsize=16, fontweight='bold')
    ax.axis('off')

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Causal Effect Strength', fontsize=12)

    report.add_figure(fig, "Causal graph showing directional influence between model components. Edge colors indicate causal effect strength.")

    # Causal effect strengths
    fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Distribution of causal effects
    effects = [G[u][v]['weight'] for u, v in G.edges()]
    axes[0].hist(effects, bins=20, alpha=0.7, edgecolor='black', color='steelblue')
    axes[0].set_xlabel('Causal Effect Strength')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Causal Effect Strengths')
    axes[0].grid(True, alpha=0.3)

    # Layer-wise causal influence
    layer_influence = {}
    for layer in range(num_layers):
        outgoing = sum([G[u][v]['weight'] for u, v in G.edges() if u.startswith(f"L{layer}")])
        layer_influence[layer] = outgoing

    layers = list(layer_influence.keys())
    influence = list(layer_influence.values())

    axes[1].bar(layers, influence, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Total Outgoing Causal Influence')
    axes[1].set_title('Layer-wise Causal Influence')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    report.add_figure(fig2, "Analysis of causal effect magnitudes and layer-wise influence patterns")

    # Create causal effects table
    edge_data = []
    for u, v in G.edges():
        edge_data.append({
            'Source': u,
            'Target': v,
            'Effect Strength': f"{G[u][v]['weight']:.3f}",
            'Type': 'Direct'
        })

    effects_df = pd.DataFrame(edge_data[:15])  # Top 15 edges
    report.add_table(effects_df, "Sample of strongest causal relationships in the network")

    # Generate reports
    html_path = report.generate_html("causal_analysis.html")
    md_path = report.generate_markdown("causal_analysis.md")

    print(f"\n✓ Generated HTML report: {html_path}")
    print(f"✓ Generated Markdown report: {md_path}")


def example_3_unified_reporter():
    """Example 3: Using UnifiedMechIntReporter for comprehensive analysis."""
    print("\n" + "="*80)
    print("Example 3: Unified Mechanistic Interpretability Reporter")
    print("="*80)

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 10)
            )

        def forward(self, x):
            return self.layers(x)

    model = SimpleModel()

    # Create synthetic data
    data = torch.randn(100, 128)

    # Configuration for analyses
    config = {
        'num_samples': 50,
        'sae': {
            'input_dim': 512,
            'hidden_dim': 1024,
            'k_active': 32
        }
    }

    # Create unified reporter
    reporter = UnifiedMechIntReporter(
        model=model,
        data=data,
        config=config,
        output_dir="./reports/example3"
    )

    # Note: This will try to run all analyses, but some may fail
    # if dependencies aren't available. That's okay for this example.
    print("\nRunning selected analyses (this may take a moment)...")

    try:
        # Run subset of analyses that don't require external modules
        results = reporter.run_selected_analyses([
            "energy_flow",
            "dynamics"
        ])

        print(f"✓ Completed {len(results)} analyses")

        # Generate comprehensive report
        print("\nGenerating unified report...")
        report = reporter.generate_report(
            title="Comprehensive Mechanistic Interpretability Analysis"
        )

        # Add custom analysis section
        report.add_section(
            "Custom Analysis: Feature Importance",
            """
This section presents a custom analysis of feature importance across the model.

We analyzed which input features have the strongest influence on model predictions
using gradient-based attribution methods.

**Top 5 Most Important Features:**
1. Feature 42: Speech amplitude modulation (importance: 0.87)
2. Feature 17: Spectral centroid (importance: 0.79)
3. Feature 91: Temporal coherence (importance: 0.71)
4. Feature 5: Phase coupling (importance: 0.68)
5. Feature 33: Cross-frequency correlation (importance: 0.64)
            """
        )

        # Generate reports
        html_path = report.generate_html("unified_report.html")
        md_path = report.generate_markdown("unified_report.md")

        print(f"\n✓ Generated HTML report: {html_path}")
        print(f"✓ Generated Markdown report: {md_path}")

    except Exception as e:
        print(f"\nNote: Some analyses may not run without full dependencies.")
        print(f"Error: {e}")
        print("\nThis is expected in a demonstration environment.")


def example_4_mlflow_integration():
    """Example 4: MLflow integration."""
    print("\n" + "="*80)
    print("Example 4: MLflow Integration")
    print("="*80)

    try:
        import mlflow

        # Create report
        report = MechIntReport(
            output_dir="./reports/example4",
            title="MLflow Integration Example"
        )

        # Add content
        report.add_section("Introduction", "This report demonstrates MLflow integration.")
        report.add_metric("Test Accuracy", 0.95, unit="%")
        report.add_metric("Inference Time", 23.5, unit="ms")

        # Create a simple plot
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        ax.plot(x, np.sin(x))
        ax.set_title("Example Plot for MLflow")

        report.add_figure(fig, "Simple sine wave")

        # Start MLflow run
        mlflow.set_experiment("mechanistic_interpretability")

        with mlflow.start_run(run_name="example_report"):
            # Export to MLflow
            report.export_to_mlflow()
            print("\n✓ Successfully logged report to MLflow")
            print(f"✓ Run ID: {mlflow.active_run().info.run_id}")

    except ImportError:
        print("\nMLflow not installed. Install with: pip install mlflow")
        print("This example would log the report to MLflow.")
    except Exception as e:
        print(f"\nNote: MLflow integration example encountered an issue: {e}")
        print("This is often due to environment setup.")


def example_5_wandb_integration():
    """Example 5: Weights & Biases integration."""
    print("\n" + "="*80)
    print("Example 5: Weights & Biases Integration")
    print("="*80)

    try:
        import wandb

        # Create report
        report = MechIntReport(
            output_dir="./reports/example5",
            title="W&B Integration Example"
        )

        # Add content
        report.add_section("Introduction", "This report demonstrates W&B integration.")
        report.add_metric("Model Size", 125, unit="MB")
        report.add_metric("GPU Memory", 8.5, unit="GB")

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        ax.plot(x, np.exp(-x/3) * np.sin(x))
        ax.set_title("Damped Oscillation")

        report.add_figure(fig, "Example damped oscillation")

        # Initialize W&B (in offline mode for example)
        wandb.init(
            project="neurofmx-interpretability",
            name="example_report",
            mode="offline"  # Use offline mode for example
        )

        # Export to W&B
        report.export_to_wandb()
        print("\n✓ Successfully logged report to W&B (offline mode)")

        wandb.finish()

    except ImportError:
        print("\nW&B not installed. Install with: pip install wandb")
        print("This example would log the report to Weights & Biases.")
    except Exception as e:
        print(f"\nNote: W&B integration example encountered an issue: {e}")
        print("This is often due to environment setup.")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("UNIFIED MECHANISTIC INTERPRETABILITY REPORTING SYSTEM")
    print("Comprehensive Examples")
    print("="*80)

    # Run examples
    example_1_basic_report()
    example_2_causal_analysis_report()
    example_3_unified_reporter()

    # Optional examples (require additional setup)
    print("\n" + "="*80)
    print("Optional Integration Examples")
    print("="*80)

    example_4_mlflow_integration()
    example_5_wandb_integration()

    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)
    print("\nGenerated reports can be found in the ./reports/ directory")
    print("Open the HTML files in your browser to view interactive reports.")


if __name__ == "__main__":
    main()
