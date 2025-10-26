"""
Unit tests for the unified mechanistic interpretability reporting system.
"""

import unittest
from pathlib import Path
import tempfile
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuros_neurofm.interpretability.reporting import (
    MechIntReport,
    ReportSection,
    ReportMetric,
    ReportTemplate
)


class TestMechIntReport(unittest.TestCase):
    """Test suite for MechIntReport class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "test_reports"

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test report initialization."""
        report = MechIntReport(self.output_dir, title="Test Report")

        self.assertEqual(report.title, "Test Report")
        self.assertEqual(len(report.sections), 0)
        self.assertEqual(len(report.metrics), 0)
        self.assertTrue(self.output_dir.exists())

    def test_add_section(self):
        """Test adding sections."""
        report = MechIntReport(self.output_dir)

        report.add_section("Introduction", "This is the introduction.")
        report.add_section("Methods", "This is the methods section.")

        self.assertEqual(len(report.sections), 2)
        self.assertEqual(report.sections[0].title, "Introduction")
        self.assertEqual(report.sections[0].content, "This is the introduction.")
        self.assertEqual(report.sections[1].title, "Methods")

    def test_add_metric(self):
        """Test adding metrics."""
        report = MechIntReport(self.output_dir)

        report.add_metric("Accuracy", 0.95, unit="%", description="Test accuracy")
        report.add_metric("Loss", 0.123)

        self.assertEqual(len(report.metrics), 2)
        self.assertEqual(report.metrics[0].name, "Accuracy")
        self.assertEqual(report.metrics[0].value, 0.95)
        self.assertEqual(report.metrics[0].unit, "%")
        self.assertEqual(report.metrics[1].name, "Loss")

    def test_add_figure(self):
        """Test adding matplotlib figures."""
        report = MechIntReport(self.output_dir)

        # Create a simple matplotlib figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        ax.set_title("Test Plot")

        report.add_figure(fig, "A test plot")

        self.assertEqual(len(report.figures), 1)
        self.assertEqual(report.figures[0]["caption"], "A test plot")
        self.assertEqual(report.figures[0]["type"], "matplotlib")

    def test_add_table(self):
        """Test adding tables."""
        report = MechIntReport(self.output_dir)

        # Create a simple DataFrame
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })

        report.add_table(df, "Test table")

        self.assertEqual(len(report.tables), 1)
        self.assertEqual(report.tables[0]["caption"], "Test table")
        self.assertTrue(isinstance(report.tables[0]["dataframe"], pd.DataFrame))

    def test_add_code(self):
        """Test adding code snippets."""
        report = MechIntReport(self.output_dir)

        code = "def test():\n    return 42"
        report.add_code(code, language="python")

        self.assertEqual(len(report.code_snippets), 1)
        self.assertEqual(report.code_snippets[0]["code"], code)
        self.assertEqual(report.code_snippets[0]["language"], "python")

    def test_generate_html(self):
        """Test HTML generation."""
        report = MechIntReport(self.output_dir, title="HTML Test")

        report.add_section("Introduction", "Test content")
        report.add_metric("Test Metric", 42, unit="units")

        html_path = report.generate_html("test.html")

        self.assertTrue(html_path.exists())
        self.assertTrue(html_path.name.endswith(".html"))

        # Check that HTML contains expected content
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        self.assertIn("HTML Test", html_content)
        self.assertIn("Introduction", html_content)
        self.assertIn("Test Metric", html_content)

    def test_generate_markdown(self):
        """Test Markdown generation."""
        report = MechIntReport(self.output_dir, title="Markdown Test")

        report.add_section("Introduction", "Test content")
        report.add_metric("Test Metric", 42, unit="units")

        md_path = report.generate_markdown("test.md")

        self.assertTrue(md_path.exists())
        self.assertTrue(md_path.name.endswith(".md"))

        # Check that Markdown contains expected content
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        self.assertIn("# Markdown Test", md_content)
        self.assertIn("## Introduction", md_content)
        self.assertIn("Test Metric", md_content)

    def test_complete_report(self):
        """Test creating a complete report with all components."""
        report = MechIntReport(self.output_dir, title="Complete Report")

        # Add section
        report.add_section(
            "Overview",
            "This is a complete test report with all components."
        )

        # Add metrics
        report.add_metric("Accuracy", 95.5, unit="%")
        report.add_metric("Loss", 0.123)
        report.add_metric("Parameters", 125000000)

        # Add figure
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 100)
        ax.plot(x, np.sin(x))
        ax.set_title("Sine Wave")
        report.add_figure(fig, "A sine wave plot")

        # Add table
        df = pd.DataFrame({
            'Layer': ['Layer 1', 'Layer 2', 'Layer 3'],
            'Units': [128, 256, 128]
        })
        report.add_table(df, "Layer configuration")

        # Add code
        code = """
def example():
    print("Hello, World!")
        """
        report.add_code(code, language="python")

        # Generate both formats
        html_path = report.generate_html("complete.html")
        md_path = report.generate_markdown("complete.md")

        # Verify both files exist
        self.assertTrue(html_path.exists())
        self.assertTrue(md_path.exists())

        # Verify HTML content
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        self.assertIn("Complete Report", html_content)
        self.assertIn("Overview", html_content)
        self.assertIn("Accuracy", html_content)
        self.assertIn("Sine Wave", html_content)

        # Verify Markdown content
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        self.assertIn("# Complete Report", md_content)
        self.assertIn("## Overview", md_content)


class TestReportSection(unittest.TestCase):
    """Test suite for ReportSection dataclass."""

    def test_creation(self):
        """Test creating a report section."""
        section = ReportSection(
            title="Test Section",
            content="Test content",
            section_type="text"
        )

        self.assertEqual(section.title, "Test Section")
        self.assertEqual(section.content, "Test content")
        self.assertEqual(section.section_type, "text")
        self.assertEqual(len(section.metadata), 0)

    def test_with_metadata(self):
        """Test section with metadata."""
        section = ReportSection(
            title="Figure Section",
            content="",
            section_type="figure",
            metadata={"figure_index": 0}
        )

        self.assertEqual(section.section_type, "figure")
        self.assertEqual(section.metadata["figure_index"], 0)


class TestReportMetric(unittest.TestCase):
    """Test suite for ReportMetric dataclass."""

    def test_creation(self):
        """Test creating a metric."""
        metric = ReportMetric(
            name="Accuracy",
            value=0.95,
            unit="%",
            description="Model accuracy"
        )

        self.assertEqual(metric.name, "Accuracy")
        self.assertEqual(metric.value, 0.95)
        self.assertEqual(metric.unit, "%")
        self.assertEqual(metric.description, "Model accuracy")

    def test_without_unit(self):
        """Test metric without unit."""
        metric = ReportMetric(name="Loss", value=0.123)

        self.assertEqual(metric.name, "Loss")
        self.assertEqual(metric.value, 0.123)
        self.assertEqual(metric.unit, "")
        self.assertEqual(metric.description, "")


class TestReportTemplate(unittest.TestCase):
    """Test suite for ReportTemplate class."""

    def test_get_analysis_template(self):
        """Test getting analysis templates."""
        template = ReportTemplate.get_analysis_template("causal_graphs")
        self.assertIsInstance(template, str)
        self.assertIn("causal-graphs", template)

        template = ReportTemplate.get_analysis_template("energy_flow")
        self.assertIsInstance(template, str)
        self.assertIn("energy-flow", template)

        # Unknown template
        template = ReportTemplate.get_analysis_template("unknown")
        self.assertIn("{{ content }}", template)

    def test_get_custom_css(self):
        """Test getting custom CSS."""
        css = ReportTemplate.get_custom_css()
        self.assertIsInstance(css, str)
        self.assertIn(".analysis-section", css)
        self.assertIn(".metric", css)


class TestReportIntegration(unittest.TestCase):
    """Integration tests for the reporting system."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "integration_tests"

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_multi_format_consistency(self):
        """Test that HTML and Markdown contain consistent information."""
        report = MechIntReport(self.output_dir, title="Consistency Test")

        report.add_section("Test Section", "Test content")
        report.add_metric("Test Metric", 42)

        html_path = report.generate_html()
        md_path = report.generate_markdown()

        # Read both files
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        # Check that key content appears in both
        self.assertIn("Consistency Test", html_content)
        self.assertIn("Consistency Test", md_content)
        self.assertIn("Test Section", html_content)
        self.assertIn("Test Section", md_content)
        self.assertIn("Test Metric", html_content)
        self.assertIn("Test Metric", md_content)

    def test_multiple_reports_same_directory(self):
        """Test generating multiple reports in the same directory."""
        report1 = MechIntReport(self.output_dir, title="Report 1")
        report1.add_section("Section 1", "Content 1")
        path1 = report1.generate_html("report1.html")

        report2 = MechIntReport(self.output_dir, title="Report 2")
        report2.add_section("Section 2", "Content 2")
        path2 = report2.generate_html("report2.html")

        # Both files should exist
        self.assertTrue(path1.exists())
        self.assertTrue(path2.exists())
        self.assertNotEqual(path1, path2)

        # Check content is different
        with open(path1, 'r', encoding='utf-8') as f:
            content1 = f.read()

        with open(path2, 'r', encoding='utf-8') as f:
            content2 = f.read()

        self.assertIn("Report 1", content1)
        self.assertIn("Report 2", content2)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMechIntReport))
    suite.addTests(loader.loadTestsFromTestCase(TestReportSection))
    suite.addTests(loader.loadTestsFromTestCase(TestReportMetric))
    suite.addTests(loader.loadTestsFromTestCase(TestReportTemplate))
    suite.addTests(loader.loadTestsFromTestCase(TestReportIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
