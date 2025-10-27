"""
Unified Reporting System for Mechanistic Interpretability Analyses.

This module provides a comprehensive reporting framework for generating
professional HTML and Markdown reports from all mech-int analyses.
"""

import base64
import io
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Template

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class ReportSection:
    """A section of a mechanistic interpretability report."""
    title: str
    content: str
    section_type: str = "text"  # text, figure, table, metric, code
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportMetric:
    """A scalar metric in a report."""
    name: str
    value: Union[float, int, str]
    unit: str = ""
    description: str = ""


class MechIntReport:
    """
    Generator for mechanistic interpretability reports.

    Supports multiple output formats (HTML, Markdown) and integrations
    with experiment tracking platforms (MLflow, W&B).
    """

    def __init__(self, output_dir: Union[str, Path], title: str = "NeuroFMX Mechanistic Interpretability Report"):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save reports
            title: Report title
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.title = title
        self.sections: List[ReportSection] = []
        self.metrics: List[ReportMetric] = []
        self.figures: List[Dict[str, Any]] = []
        self.tables: List[Dict[str, Any]] = []
        self.code_snippets: List[Dict[str, Any]] = []

        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "title": title,
        }

    def add_section(self, title: str, content: str) -> None:
        """
        Add a markdown section to the report.

        Args:
            title: Section title
            content: Section content (markdown supported)
        """
        section = ReportSection(
            title=title,
            content=content,
            section_type="text"
        )
        self.sections.append(section)

    def add_figure(
        self,
        fig: Union[plt.Figure, go.Figure],
        caption: str,
        name: Optional[str] = None
    ) -> None:
        """
        Add a matplotlib or plotly figure to the report.

        Args:
            fig: Matplotlib or Plotly figure
            caption: Figure caption
            name: Optional figure name (auto-generated if None)
        """
        if name is None:
            name = f"figure_{len(self.figures) + 1}"

        figure_data = {
            "name": name,
            "caption": caption,
            "figure": fig,
            "type": "matplotlib" if isinstance(fig, plt.Figure) else "plotly"
        }
        self.figures.append(figure_data)

        # Also add as section
        section = ReportSection(
            title=caption,
            content="",
            section_type="figure",
            metadata={"figure_index": len(self.figures) - 1}
        )
        self.sections.append(section)

    def add_table(
        self,
        df: pd.DataFrame,
        caption: str,
        name: Optional[str] = None
    ) -> None:
        """
        Add a pandas DataFrame table to the report.

        Args:
            df: DataFrame to add
            caption: Table caption
            name: Optional table name (auto-generated if None)
        """
        if name is None:
            name = f"table_{len(self.tables) + 1}"

        table_data = {
            "name": name,
            "caption": caption,
            "dataframe": df
        }
        self.tables.append(table_data)

        # Also add as section
        section = ReportSection(
            title=caption,
            content="",
            section_type="table",
            metadata={"table_index": len(self.tables) - 1}
        )
        self.sections.append(section)

    def add_metric(
        self,
        name: str,
        value: Union[float, int, str],
        unit: str = "",
        description: str = ""
    ) -> None:
        """
        Add a scalar metric to the report.

        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            description: Optional description
        """
        metric = ReportMetric(
            name=name,
            value=value,
            unit=unit,
            description=description
        )
        self.metrics.append(metric)

    def add_code(self, code: str, language: str = "python") -> None:
        """
        Add a code snippet to the report.

        Args:
            code: Code content
            language: Programming language for syntax highlighting
        """
        code_data = {
            "code": code,
            "language": language
        }
        self.code_snippets.append(code_data)

        # Also add as section
        section = ReportSection(
            title=f"Code ({language})",
            content="",
            section_type="code",
            metadata={"code_index": len(self.code_snippets) - 1}
        )
        self.sections.append(section)

    def _figure_to_base64(self, fig: Union[plt.Figure, go.Figure]) -> str:
        """Convert figure to base64 encoded string."""
        if isinstance(fig, plt.Figure):
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            plt.close(fig)
            return f"data:image/png;base64,{img_str}"
        else:  # Plotly figure
            buf = io.BytesIO()
            fig.write_image(buf, format='png', width=800, height=600)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            return f"data:image/png;base64,{img_str}"

    def _figure_to_html(self, fig: Union[plt.Figure, go.Figure]) -> str:
        """Convert figure to HTML (interactive for Plotly)."""
        if isinstance(fig, plt.Figure):
            img_b64 = self._figure_to_base64(fig)
            return f'<img src="{img_b64}" style="max-width: 100%; height: auto;" />'
        else:  # Plotly figure - embed interactive plot
            return pio.to_html(fig, include_plotlyjs='cdn', full_html=False)

    def generate_html(self, filename: str = "report.html") -> Path:
        """
        Generate an HTML report with embedded figures.

        Args:
            filename: Output filename

        Returns:
            Path to generated HTML file
        """
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <script src="https://cdn.plot.ly/plotly-2.18.0.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/github.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }

        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }

        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        header .timestamp {
            opacity: 0.9;
            font-size: 0.9em;
        }

        nav {
            background: #2d3748;
            padding: 15px 40px;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }

        nav a {
            color: white;
            text-decoration: none;
            margin-right: 20px;
            padding: 5px 10px;
            border-radius: 4px;
            transition: background 0.3s;
        }

        nav a:hover {
            background: rgba(255,255,255,0.1);
        }

        .content {
            padding: 40px;
        }

        .section {
            margin: 30px 0;
            padding: 25px;
            border-left: 4px solid #667eea;
            background: #f8f9fa;
            border-radius: 4px;
        }

        .section h2 {
            color: #2d3748;
            margin-bottom: 15px;
            font-size: 1.8em;
        }

        .section h3 {
            color: #4a5568;
            margin: 20px 0 10px 0;
            font-size: 1.4em;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }

        .metric {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            text-align: center;
            border-top: 3px solid #667eea;
        }

        .metric-name {
            font-size: 0.9em;
            color: #718096;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #2d3748;
        }

        .metric-unit {
            font-size: 0.8em;
            color: #a0aec0;
            margin-left: 5px;
        }

        .metric-description {
            font-size: 0.85em;
            color: #718096;
            margin-top: 8px;
        }

        figure {
            margin: 30px 0;
            text-align: center;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }

        figure img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }

        figcaption {
            margin-top: 15px;
            font-style: italic;
            color: #718096;
            font-size: 0.95em;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            border-radius: 8px;
            overflow: hidden;
        }

        th {
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }

        td {
            padding: 12px;
            border-bottom: 1px solid #e2e8f0;
        }

        tr:last-child td {
            border-bottom: none;
        }

        tr:hover {
            background: #f7fafc;
        }

        .code-block {
            background: #2d3748;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            overflow-x: auto;
        }

        .code-block pre {
            margin: 0;
        }

        .code-block code {
            color: #e2e8f0;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }

        footer {
            background: #2d3748;
            color: white;
            text-align: center;
            padding: 20px;
            margin-top: 40px;
        }

        @media print {
            nav {
                display: none;
            }

            .section {
                break-inside: avoid;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{{ title }}</h1>
            <div class="timestamp">Generated on {{ timestamp }}</div>
        </header>

        <nav id="nav">
            <a href="#summary">Summary</a>
            {% for section in sections %}
            {% if section.section_type == "text" %}
            <a href="#{{ section.title|lower|replace(' ', '-') }}">{{ section.title }}</a>
            {% endif %}
            {% endfor %}
        </nav>

        <div class="content">
            <!-- Summary Section -->
            <div class="section" id="summary">
                <h2>Summary</h2>

                {% if metrics %}
                <h3>Key Metrics</h3>
                <div class="metrics-grid">
                    {% for metric in metrics %}
                    <div class="metric">
                        <div class="metric-name">{{ metric.name }}</div>
                        <div class="metric-value">
                            {{ metric.value }}
                            <span class="metric-unit">{{ metric.unit }}</span>
                        </div>
                        {% if metric.description %}
                        <div class="metric-description">{{ metric.description }}</div>
                        {% endif %}
                    </div>
                    {% endfor %}
                </div>
                {% endif %}
            </div>

            <!-- Report Sections -->
            {% for section in sections %}
            <div class="section" id="{{ section.title|lower|replace(' ', '-') }}">
                <h2>{{ section.title }}</h2>

                {% if section.section_type == "text" %}
                    <div>{{ section.content|safe }}</div>

                {% elif section.section_type == "figure" %}
                    <figure>
                        {{ figures_html[section.metadata.figure_index]|safe }}
                        <figcaption>{{ figures[section.metadata.figure_index].caption }}</figcaption>
                    </figure>

                {% elif section.section_type == "table" %}
                    <figure>
                        {{ tables_html[section.metadata.table_index]|safe }}
                        <figcaption>{{ tables[section.metadata.table_index].caption }}</figcaption>
                    </figure>

                {% elif section.section_type == "code" %}
                    <div class="code-block">
                        <pre><code class="language-{{ code_snippets[section.metadata.code_index].language }}">{{ code_snippets[section.metadata.code_index].code }}</code></pre>
                    </div>
                {% endif %}
            </div>
            {% endfor %}
        </div>

        <footer>
            <p>Generated by NeuroFMX Mechanistic Interpretability Suite</p>
            <p>&copy; {{ year }} neurOS-v1</p>
        </footer>
    </div>

    <script>
        // Initialize syntax highlighting
        hljs.highlightAll();

        // Smooth scrolling for navigation
        document.querySelectorAll('nav a').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            });
        });
    </script>
</body>
</html>
        """

        # Prepare figures as HTML
        figures_html = []
        for fig_data in self.figures:
            html = self._figure_to_html(fig_data["figure"])
            figures_html.append(html)

        # Prepare tables as HTML
        tables_html = []
        for table_data in self.tables:
            df = table_data["dataframe"]
            html = df.to_html(classes='dataframe', border=0, index=True)
            tables_html.append(html)

        # Render template
        template = Template(template_str)
        html_content = template.render(
            title=self.title,
            timestamp=self.metadata["created_at"],
            year=datetime.now().year,
            sections=self.sections,
            metrics=self.metrics,
            figures=self.figures,
            figures_html=figures_html,
            tables=self.tables,
            tables_html=tables_html,
            code_snippets=self.code_snippets
        )

        # Save to file
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_path

    def generate_markdown(self, filename: str = "report.md") -> Path:
        """
        Generate a Markdown report.

        Args:
            filename: Output filename

        Returns:
            Path to generated Markdown file
        """
        md_lines = [
            f"# {self.title}\n",
            f"*Generated on {self.metadata['created_at']}*\n",
            "\n---\n",
        ]

        # Summary section
        md_lines.append("\n## Summary\n")

        if self.metrics:
            md_lines.append("\n### Key Metrics\n")
            for metric in self.metrics:
                value_str = f"{metric.value} {metric.unit}".strip()
                md_lines.append(f"- **{metric.name}**: {value_str}")
                if metric.description:
                    md_lines.append(f"  - {metric.description}")
            md_lines.append("\n")

        # Add all sections
        for idx, section in enumerate(self.sections):
            md_lines.append(f"\n## {section.title}\n")

            if section.section_type == "text":
                md_lines.append(f"{section.content}\n")

            elif section.section_type == "figure":
                fig_idx = section.metadata["figure_index"]
                fig_data = self.figures[fig_idx]
                fig_name = fig_data["name"]

                # Save figure to file
                fig_path = self.output_dir / f"{fig_name}.png"
                if isinstance(fig_data["figure"], plt.Figure):
                    fig_data["figure"].savefig(fig_path, dpi=150, bbox_inches='tight')
                    plt.close(fig_data["figure"])
                else:
                    fig_data["figure"].write_image(fig_path, width=800, height=600)

                md_lines.append(f"![{fig_data['caption']}]({fig_path.name})\n")
                md_lines.append(f"*{fig_data['caption']}*\n")

            elif section.section_type == "table":
                table_idx = section.metadata["table_index"]
                table_data = self.tables[table_idx]
                df = table_data["dataframe"]

                md_lines.append(df.to_markdown())
                md_lines.append(f"\n*{table_data['caption']}*\n")

            elif section.section_type == "code":
                code_idx = section.metadata["code_index"]
                code_data = self.code_snippets[code_idx]

                md_lines.append(f"```{code_data['language']}")
                md_lines.append(code_data['code'])
                md_lines.append("```\n")

        # Save to file
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))

        return output_path

    def export_to_mlflow(self, mlflow_client: Any = None, run_id: Optional[str] = None) -> None:
        """
        Log report to MLflow.

        Args:
            mlflow_client: MLflow client (uses mlflow if None)
            run_id: MLflow run ID (uses active run if None)
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is not installed. Install with: pip install mlflow")

        client = mlflow_client or mlflow

        # Log metrics
        for metric in self.metrics:
            if isinstance(metric.value, (int, float)):
                client.log_metric(metric.name, metric.value)

        # Log figures
        for fig_data in self.figures:
            fig_name = fig_data["name"]
            fig_path = self.output_dir / f"{fig_name}.png"

            if isinstance(fig_data["figure"], plt.Figure):
                fig_data["figure"].savefig(fig_path, dpi=150, bbox_inches='tight')
                plt.close(fig_data["figure"])
            else:
                fig_data["figure"].write_image(fig_path, width=800, height=600)

            client.log_artifact(str(fig_path))

        # Log HTML report
        html_path = self.generate_html()
        client.log_artifact(str(html_path))

        # Log markdown report
        md_path = self.generate_markdown()
        client.log_artifact(str(md_path))

    def export_to_wandb(self, wandb_run: Any = None) -> None:
        """
        Log report to Weights & Biases.

        Args:
            wandb_run: W&B run object (uses wandb.run if None)
        """
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Install with: pip install wandb")

        run = wandb_run or wandb.run
        if run is None:
            raise ValueError("No active W&B run. Initialize with wandb.init()")

        # Log metrics
        metrics_dict = {}
        for metric in self.metrics:
            if isinstance(metric.value, (int, float)):
                metrics_dict[metric.name] = metric.value
        run.log(metrics_dict)

        # Log figures
        for fig_data in self.figures:
            if isinstance(fig_data["figure"], plt.Figure):
                run.log({fig_data["name"]: wandb.Image(fig_data["figure"])})
                plt.close(fig_data["figure"])
            else:
                # Convert Plotly to image
                fig_path = self.output_dir / f"{fig_data['name']}.png"
                fig_data["figure"].write_image(fig_path, width=800, height=600)
                run.log({fig_data["name"]: wandb.Image(str(fig_path))})

        # Log tables
        for table_data in self.tables:
            run.log({table_data["name"]: wandb.Table(dataframe=table_data["dataframe"])})

        # Log HTML report
        html_path = self.generate_html()
        run.log_artifact(str(html_path), type="report")


class UnifiedMechIntReporter:
    """
    Unified reporter that runs all mechanistic interpretability analyses
    and generates a comprehensive report.
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Union[str, Path] = "reports"
    ):
        """
        Initialize unified reporter.

        Args:
            model: Model to analyze
            data: Data for analysis
            config: Configuration dict for analyses
            output_dir: Directory to save reports
        """
        self.model = model
        self.data = data
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.analyses_results = {}

    def run_all_analyses(self) -> Dict[str, Any]:
        """
        Run all enabled mechanistic interpretability analyses.

        Returns:
            Dictionary of analysis results
        """
        analyses = [
            "causal_graphs",
            "energy_flow",
            "topology",
            "sae_features",
            "alignment",
            "dynamics",
            "counterfactuals",
            "attribution"
        ]

        return self.run_selected_analyses(analyses)

    def run_selected_analyses(self, analyses: List[str]) -> Dict[str, Any]:
        """
        Run specific analyses.

        Args:
            analyses: List of analysis names to run

        Returns:
            Dictionary of analysis results
        """
        results = {}

        for analysis_name in analyses:
            try:
                if analysis_name == "causal_graphs":
                    results[analysis_name] = self._run_causal_graphs()
                elif analysis_name == "energy_flow":
                    results[analysis_name] = self._run_energy_flow()
                elif analysis_name == "topology":
                    results[analysis_name] = self._run_topology()
                elif analysis_name == "sae_features":
                    results[analysis_name] = self._run_sae_features()
                elif analysis_name == "alignment":
                    results[analysis_name] = self._run_alignment()
                elif analysis_name == "dynamics":
                    results[analysis_name] = self._run_dynamics()
                elif analysis_name == "counterfactuals":
                    results[analysis_name] = self._run_counterfactuals()
                elif analysis_name == "attribution":
                    results[analysis_name] = self._run_attribution()
                else:
                    print(f"Unknown analysis: {analysis_name}")
            except Exception as e:
                print(f"Error running {analysis_name}: {e}")
                results[analysis_name] = {"error": str(e)}

        self.analyses_results = results
        return results

    def _run_causal_graphs(self) -> Dict[str, Any]:
        """Run causal graph analysis."""
        from .causal_graphs import CausalGraphAnalyzer

        analyzer = CausalGraphAnalyzer(self.model)

        # Generate synthetic data if needed
        if hasattr(self.data, 'shape'):
            data = self.data[:self.config.get('num_samples', 100)]
        else:
            data = self.data

        graph = analyzer.build_causal_graph(data)
        effects = analyzer.compute_causal_effects(graph, data)

        return {
            "graph": graph,
            "effects": effects,
            "num_nodes": len(graph.nodes()),
            "num_edges": len(graph.edges())
        }

    def _run_energy_flow(self) -> Dict[str, Any]:
        """Run energy flow analysis."""
        from .energy_flow import EnergyFlowAnalyzer

        analyzer = EnergyFlowAnalyzer(self.model)

        if hasattr(self.data, 'shape'):
            data = self.data[:1]  # Single sample
        else:
            data = self.data

        flow = analyzer.compute_energy_flow(data)

        return {
            "flow": flow,
            "total_energy": float(np.sum([f["energy"] for f in flow.values()])),
            "num_layers": len(flow)
        }

    def _run_topology(self) -> Dict[str, Any]:
        """Run topological analysis."""
        from .topology import TopologyAnalyzer

        analyzer = TopologyAnalyzer(self.model)

        if hasattr(self.data, 'shape'):
            data = self.data[:self.config.get('num_samples', 50)]
        else:
            data = self.data

        features = analyzer.extract_features(data)
        topology = analyzer.compute_persistent_homology(features)

        return {
            "topology": topology,
            "betti_numbers": topology.get("betti_numbers", []),
            "num_features": features.shape[0] if hasattr(features, 'shape') else len(features)
        }

    def _run_sae_features(self) -> Dict[str, Any]:
        """Run SAE feature analysis."""
        from .sae_features import SAEFeatureAnalyzer

        # This requires a trained SAE
        config = self.config.get('sae', {})
        analyzer = SAEFeatureAnalyzer(
            input_dim=config.get('input_dim', 512),
            hidden_dim=config.get('hidden_dim', 1024),
            k_active=config.get('k_active', 32)
        )

        if hasattr(self.data, 'shape'):
            data = self.data[:self.config.get('num_samples', 100)]
        else:
            data = self.data

        features = analyzer.extract_features(data)
        interpretations = analyzer.interpret_features(features)

        return {
            "features": features,
            "interpretations": interpretations,
            "num_active": int(np.sum(features > 0)),
            "sparsity": float(np.mean(features == 0))
        }

    def _run_alignment(self) -> Dict[str, Any]:
        """Run representational alignment analysis."""
        from .alignment import AlignmentAnalyzer

        analyzer = AlignmentAnalyzer(self.model)

        if hasattr(self.data, 'shape'):
            data = self.data[:self.config.get('num_samples', 100)]
        else:
            data = self.data

        alignment_scores = analyzer.compute_alignment(data)

        return {
            "alignment_scores": alignment_scores,
            "mean_alignment": float(np.mean(list(alignment_scores.values()))),
            "num_layers": len(alignment_scores)
        }

    def _run_dynamics(self) -> Dict[str, Any]:
        """Run dynamics analysis."""
        from .dynamics import DynamicsAnalyzer

        analyzer = DynamicsAnalyzer(self.model)

        if hasattr(self.data, 'shape'):
            data = self.data[:1]  # Single trajectory
        else:
            data = self.data

        trajectory = analyzer.analyze_trajectory(data)

        return {
            "trajectory": trajectory,
            "num_steps": len(trajectory),
            "final_state_norm": float(np.linalg.norm(trajectory[-1]))
        }

    def _run_counterfactuals(self) -> Dict[str, Any]:
        """Run counterfactual analysis."""
        from .counterfactuals import CounterfactualAnalyzer

        analyzer = CounterfactualAnalyzer(self.model)

        if hasattr(self.data, 'shape'):
            data = self.data[:self.config.get('num_samples', 10)]
        else:
            data = self.data

        counterfactuals = analyzer.generate_counterfactuals(data)

        return {
            "counterfactuals": counterfactuals,
            "num_generated": len(counterfactuals),
            "avg_distance": float(np.mean([cf["distance"] for cf in counterfactuals]))
        }

    def _run_attribution(self) -> Dict[str, Any]:
        """Run attribution analysis."""
        from .attribution import AttributionAnalyzer

        analyzer = AttributionAnalyzer(self.model)

        if hasattr(self.data, 'shape'):
            data = self.data[:self.config.get('num_samples', 10)]
        else:
            data = self.data

        attributions = analyzer.compute_attributions(data)

        return {
            "attributions": attributions,
            "num_samples": len(attributions),
            "top_features": attributions.get("top_features", [])
        }

    def generate_report(self, title: Optional[str] = None) -> MechIntReport:
        """
        Generate a unified mechanistic interpretability report.

        Args:
            title: Optional report title

        Returns:
            MechIntReport object
        """
        if not self.analyses_results:
            self.run_all_analyses()

        if title is None:
            title = f"NeuroFMX Mechanistic Interpretability Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        report = MechIntReport(self.output_dir, title=title)

        # Add overview section
        report.add_section(
            "Overview",
            f"""
This report presents a comprehensive mechanistic interpretability analysis of the model.
The analysis was conducted on {datetime.now().strftime('%Y-%m-%d at %H:%M')}.

**Analyses Performed:**
{chr(10).join(f'- {name.replace("_", " ").title()}' for name in self.analyses_results.keys())}
            """
        )

        # Add sections for each analysis
        for analysis_name, results in self.analyses_results.items():
            if "error" in results:
                report.add_section(
                    analysis_name.replace("_", " ").title(),
                    f"**Error:** {results['error']}"
                )
                continue

            # Generate analysis-specific report section
            self._add_analysis_section(report, analysis_name, results)

        # Add summary metrics
        self._add_summary_metrics(report)

        return report

    def _add_analysis_section(
        self,
        report: MechIntReport,
        analysis_name: str,
        results: Dict[str, Any]
    ) -> None:
        """Add analysis-specific section to report."""
        title = analysis_name.replace("_", " ").title()

        if analysis_name == "causal_graphs":
            report.add_section(
                title,
                f"""
This section presents the causal structure discovered in the model.

**Graph Statistics:**
- Nodes: {results.get('num_nodes', 'N/A')}
- Edges: {results.get('num_edges', 'N/A')}
                """
            )

            # Create visualization if networkx is available
            try:
                import networkx as nx
                graph = results["graph"]

                fig, ax = plt.subplots(figsize=(12, 8))
                pos = nx.spring_layout(graph)
                nx.draw(
                    graph, pos, ax=ax,
                    with_labels=True,
                    node_color='lightblue',
                    node_size=500,
                    font_size=8,
                    arrows=True
                )
                ax.set_title("Causal Graph Structure")

                report.add_figure(fig, "Causal graph showing dependencies between model components")
            except Exception as e:
                print(f"Could not create causal graph visualization: {e}")

        elif analysis_name == "energy_flow":
            report.add_section(
                title,
                f"""
This section analyzes the flow of information energy through the model.

**Energy Statistics:**
- Total Energy: {results.get('total_energy', 'N/A'):.4f}
- Layers Analyzed: {results.get('num_layers', 'N/A')}
                """
            )

            # Create energy flow visualization
            try:
                flow = results["flow"]
                layers = list(flow.keys())
                energies = [flow[layer]["energy"] for layer in layers]

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(layers, energies, marker='o', linewidth=2)
                ax.set_xlabel("Layer")
                ax.set_ylabel("Energy")
                ax.set_title("Energy Flow Across Layers")
                ax.grid(True, alpha=0.3)

                report.add_figure(fig, "Energy flow showing information propagation through layers")
            except Exception as e:
                print(f"Could not create energy flow visualization: {e}")

        elif analysis_name == "topology":
            report.add_section(
                title,
                f"""
This section presents the topological structure of the representation space.

**Topology Statistics:**
- Betti Numbers: {results.get('betti_numbers', 'N/A')}
- Features Analyzed: {results.get('num_features', 'N/A')}
                """
            )

        elif analysis_name == "sae_features":
            report.add_section(
                title,
                f"""
This section analyzes sparse autoencoder features and their interpretations.

**Feature Statistics:**
- Active Features: {results.get('num_active', 'N/A')}
- Sparsity: {results.get('sparsity', 'N/A'):.4f}
                """
            )

        elif analysis_name == "alignment":
            report.add_section(
                title,
                f"""
This section measures alignment between model representations and target representations.

**Alignment Statistics:**
- Mean Alignment: {results.get('mean_alignment', 'N/A'):.4f}
- Layers Analyzed: {results.get('num_layers', 'N/A')}
                """
            )

        elif analysis_name == "dynamics":
            report.add_section(
                title,
                f"""
This section analyzes the dynamical evolution of representations.

**Dynamics Statistics:**
- Trajectory Steps: {results.get('num_steps', 'N/A')}
- Final State Norm: {results.get('final_state_norm', 'N/A'):.4f}
                """
            )

        elif analysis_name == "counterfactuals":
            report.add_section(
                title,
                f"""
This section presents counterfactual examples and their analysis.

**Counterfactual Statistics:**
- Generated: {results.get('num_generated', 'N/A')}
- Avg Distance: {results.get('avg_distance', 'N/A'):.4f}
                """
            )

        elif analysis_name == "attribution":
            report.add_section(
                title,
                f"""
This section shows feature attributions and importance scores.

**Attribution Statistics:**
- Samples Analyzed: {results.get('num_samples', 'N/A')}
                """
            )

    def _add_summary_metrics(self, report: MechIntReport) -> None:
        """Add summary metrics from all analyses."""
        for analysis_name, results in self.analyses_results.items():
            if "error" in results:
                continue

            # Extract key metrics
            if analysis_name == "causal_graphs":
                report.add_metric("Causal Nodes", results.get('num_nodes', 0))
                report.add_metric("Causal Edges", results.get('num_edges', 0))

            elif analysis_name == "energy_flow":
                report.add_metric(
                    "Total Energy",
                    results.get('total_energy', 0.0),
                    description="Total information energy in the model"
                )

            elif analysis_name == "sae_features":
                report.add_metric(
                    "Feature Sparsity",
                    results.get('sparsity', 0.0),
                    description="Proportion of inactive features"
                )

            elif analysis_name == "alignment":
                report.add_metric(
                    "Mean Alignment",
                    results.get('mean_alignment', 0.0),
                    description="Average alignment across layers"
                )


class ReportTemplate:
    """
    Templates for different analysis types.
    Provides HTML/CSS styling and layout templates.
    """

    @staticmethod
    def get_analysis_template(analysis_type: str) -> str:
        """
        Get HTML template for specific analysis type.

        Args:
            analysis_type: Type of analysis

        Returns:
            HTML template string
        """
        templates = {
            "causal_graphs": """
                <div class="analysis-section causal-graphs">
                    <h3>Causal Graph Analysis</h3>
                    <p>{{ description }}</p>
                    <div class="graph-container">
                        {{ graph_visualization }}
                    </div>
                    <div class="metrics">
                        <div class="metric">
                            <span class="label">Nodes:</span>
                            <span class="value">{{ num_nodes }}</span>
                        </div>
                        <div class="metric">
                            <span class="label">Edges:</span>
                            <span class="value">{{ num_edges }}</span>
                        </div>
                    </div>
                </div>
            """,

            "energy_flow": """
                <div class="analysis-section energy-flow">
                    <h3>Energy Flow Analysis</h3>
                    <p>{{ description }}</p>
                    <div class="chart-container">
                        {{ energy_chart }}
                    </div>
                    <div class="metrics">
                        <div class="metric">
                            <span class="label">Total Energy:</span>
                            <span class="value">{{ total_energy }}</span>
                        </div>
                    </div>
                </div>
            """,

            "topology": """
                <div class="analysis-section topology">
                    <h3>Topological Analysis</h3>
                    <p>{{ description }}</p>
                    <div class="topology-viz">
                        {{ topology_visualization }}
                    </div>
                    <div class="betti-numbers">
                        <h4>Betti Numbers:</h4>
                        {{ betti_numbers }}
                    </div>
                </div>
            """
        }

        return templates.get(analysis_type, "<div>{{ content }}</div>")

    @staticmethod
    def get_custom_css() -> str:
        """Get custom CSS for enhanced styling."""
        return """
        .analysis-section {
            margin: 30px 0;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .graph-container, .chart-container, .topology-viz {
            margin: 20px 0;
            text-align: center;
        }

        .metrics {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            margin-top: 20px;
        }

        .metric {
            margin: 10px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 4px;
        }

        .metric .label {
            font-weight: bold;
            color: #718096;
            margin-right: 8px;
        }

        .metric .value {
            font-size: 1.2em;
            color: #2d3748;
        }
        """


# Example usage
if __name__ == "__main__":
    # Example 1: Basic report creation
    print("Example 1: Creating a basic report")

    report = MechIntReport(output_dir="./example_reports")

    # Add sections
    report.add_section(
        "Introduction",
        "This is a demonstration of the unified reporting system."
    )

    # Add metrics
    report.add_metric("Accuracy", 0.95, unit="%", description="Model accuracy on test set")
    report.add_metric("Loss", 0.123, description="Final training loss")

    # Add a figure
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x), label='sin(x)')
    ax.plot(x, np.cos(x), label='cos(x)')
    ax.legend()
    ax.set_title("Example Plot")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    report.add_figure(fig, "Trigonometric functions demonstration")

    # Add a table
    df = pd.DataFrame({
        'Layer': [f'Layer {i}' for i in range(5)],
        'Parameters': [1000, 2000, 3000, 2000, 1000],
        'Activation': ['ReLU', 'ReLU', 'ReLU', 'ReLU', 'Softmax']
    })

    report.add_table(df, "Model architecture summary")

    # Add code snippet
    code = """
def example_function():
    print("Hello, World!")
    return 42
    """

    report.add_code(code, language="python")

    # Generate reports
    html_path = report.generate_html()
    md_path = report.generate_markdown()

    print(f"Generated HTML report: {html_path}")
    print(f"Generated Markdown report: {md_path}")

    print("\nExample completed successfully!")
