"""
Tests for the neurOS command-line interface.

Tests all CLI commands including run, benchmark, train, model registry,
dashboard, demo generation, and constellation demo.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pandas as pd
import pytest

from neuros.cli import _parse_args, main


class TestCLIParsing:
    """Test CLI argument parsing."""

    def test_parse_run_command(self):
        """Test parsing 'run' command."""
        with patch.object(sys, 'argv', ['neuros', 'run']):
            args = _parse_args()
            assert args.command == 'run'
            assert args.duration == 5.0

    def test_parse_run_with_duration(self):
        """Test parsing 'run' command with custom duration."""
        with patch.object(sys, 'argv', ['neuros', 'run', '--duration', '10.0']):
            args = _parse_args()
            assert args.command == 'run'
            assert args.duration == 10.0

    def test_parse_benchmark_command(self):
        """Test parsing 'benchmark' command."""
        with patch.object(sys, 'argv', ['neuros', 'benchmark']):
            args = _parse_args()
            assert args.command == 'benchmark'
            assert args.duration == 10.0
            assert args.report is None

    def test_parse_benchmark_with_report(self):
        """Test parsing 'benchmark' with report output."""
        with patch.object(sys, 'argv', ['neuros', 'benchmark', '--report', 'bench.json']):
            args = _parse_args()
            assert args.command == 'benchmark'
            assert args.report == 'bench.json'

    def test_parse_train_command(self):
        """Test parsing 'train' command."""
        with patch.object(sys, 'argv', ['neuros', 'train', '--csv', 'data.csv']):
            args = _parse_args()
            assert args.command == 'train'
            assert args.csv == 'data.csv'

    def test_parse_save_model_command(self):
        """Test parsing 'save-model' command."""
        with patch.object(sys, 'argv', ['neuros', 'save-model', '--model-file', 'model.pkl', '--name', 'test_model']):
            args = _parse_args()
            assert args.command == 'save-model'
            assert args.model_file == 'model.pkl'
            assert args.name == 'test_model'

    def test_parse_save_model_with_metadata(self):
        """Test parsing 'save-model' with optional metadata."""
        with patch.object(sys, 'argv', [
            'neuros', 'save-model',
            '--model-file', 'model.pkl',
            '--name', 'test_model',
            '--version', 'v1.0',
            '--tags', 'eeg', 'motor-imagery',
            '--accuracy', '0.85'
        ]):
            args = _parse_args()
            assert args.version == 'v1.0'
            assert args.tags == ['eeg', 'motor-imagery']
            assert args.accuracy == 0.85

    def test_parse_load_model_command(self):
        """Test parsing 'load-model' command."""
        with patch.object(sys, 'argv', ['neuros', 'load-model', '--name', 'test_model']):
            args = _parse_args()
            assert args.command == 'load-model'
            assert args.name == 'test_model'

    def test_parse_list_models_command(self):
        """Test parsing 'list-models' command."""
        with patch.object(sys, 'argv', ['neuros', 'list-models']):
            args = _parse_args()
            assert args.command == 'list-models'
            assert args.format == 'table'

    def test_parse_list_models_with_filters(self):
        """Test parsing 'list-models' with filters."""
        with patch.object(sys, 'argv', [
            'neuros', 'list-models',
            '--filter', 'eeg',
            '--tags', 'motor', 'imagery',
            '--format', 'json'
        ]):
            args = _parse_args()
            assert args.filter == 'eeg'
            assert args.tags == ['motor', 'imagery']
            assert args.format == 'json'

    def test_parse_dashboard_command(self):
        """Test parsing 'dashboard' command."""
        with patch.object(sys, 'argv', ['neuros', 'dashboard']):
            args = _parse_args()
            assert args.command == 'dashboard'

    def test_parse_demo_command(self):
        """Test parsing 'demo' command."""
        with patch.object(sys, 'argv', ['neuros', 'demo', '--task', '2-class motor imagery']):
            args = _parse_args()
            assert args.command == 'demo'
            assert args.task == '2-class motor imagery'
            assert args.duration == 3.0
            assert args.output_dir == 'notebooks'

    def test_parse_run_tasks_command(self):
        """Test parsing 'run-tasks' command."""
        with patch.object(sys, 'argv', [
            'neuros', 'run-tasks',
            '--tasks', 'motor imagery', 'emotion recognition'
        ]):
            args = _parse_args()
            assert args.command == 'run-tasks'
            assert args.tasks == ['motor imagery', 'emotion recognition']

    def test_parse_serve_command(self):
        """Test parsing 'serve' command."""
        with patch.object(sys, 'argv', ['neuros', 'serve']):
            args = _parse_args()
            assert args.command == 'serve'
            assert args.host == '127.0.0.1'
            assert args.port == 8000

    def test_parse_serve_with_host_port(self):
        """Test parsing 'serve' with custom host and port."""
        with patch.object(sys, 'argv', ['neuros', 'serve', '--host', '0.0.0.0', '--port', '9000']):
            args = _parse_args()
            assert args.host == '0.0.0.0'
            assert args.port == 9000

    def test_parse_constellation_command(self):
        """Test parsing 'constellation' command."""
        with patch.object(sys, 'argv', ['neuros', 'constellation']):
            args = _parse_args()
            assert args.command == 'constellation'
            assert args.duration == 10.0
            assert args.output_dir == '/tmp/constellation_demo'

    def test_parse_constellation_with_options(self):
        """Test parsing 'constellation' with all options."""
        with patch.object(sys, 'argv', [
            'neuros', 'constellation',
            '--duration', '20.0',
            '--output-dir', '/tmp/custom',
            '--subject-id', 'sub001',
            '--session-id', 'ses001',
            '--fault-injection',
            '--kafka-bootstrap', 'kafka:9092',
            '--topic-prefix', 'aligned',
            '--no-kafka'
        ]):
            args = _parse_args()
            assert args.duration == 20.0
            assert args.output_dir == '/tmp/custom'
            assert args.subject_id == 'sub001'
            assert args.session_id == 'ses001'
            assert args.fault_injection is True
            assert args.kafka_bootstrap == 'kafka:9092'
            assert args.topic_prefix == 'aligned'
            assert args.no_kafka is True


class TestCLICommands:
    """Test CLI command execution."""

    @patch('neuros.cli.asyncio.run')
    @patch('neuros.cli.Pipeline')
    @patch('builtins.print')
    def test_run_command(self, mock_print, mock_pipeline_class, mock_asyncio_run):
        """Test 'run' command execution."""
        # Setup mocks
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_asyncio_run.return_value = {'accuracy': 0.75, 'latency_ms': 10.5}

        with patch.object(sys, 'argv', ['neuros', 'run', '--duration', '2.0']):
            main()

        # Verify pipeline was created and trained
        assert mock_pipeline.train.called
        mock_asyncio_run.assert_called_once()

        # Verify output was printed
        assert mock_print.called

    @patch('neuros.cli.asyncio.run')
    @patch('neuros.cli.run_benchmark')
    @patch('builtins.print')
    def test_benchmark_command_stdout(self, mock_print, mock_benchmark, mock_asyncio_run):
        """Test 'benchmark' command with stdout output."""
        mock_asyncio_run.return_value = {'latency': 5.0, 'throughput': 100.0}

        with patch.object(sys, 'argv', ['neuros', 'benchmark']):
            main()

        assert mock_print.called

    @patch('neuros.cli.asyncio.run')
    @patch('neuros.cli.run_benchmark')
    def test_benchmark_command_file_output(self, mock_benchmark, mock_asyncio_run):
        """Test 'benchmark' command with file output."""
        mock_asyncio_run.return_value = {'latency': 5.0, 'throughput': 100.0}

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / 'benchmark.json'

            with patch.object(sys, 'argv', ['neuros', 'benchmark', '--report', str(report_path)]):
                main()

            assert report_path.exists()
            data = json.loads(report_path.read_text())
            assert 'latency' in data

    @patch('builtins.print')
    def test_train_command(self, mock_print):
        """Test 'train' command execution."""
        # Create sample CSV data
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / 'data.csv'
            df = pd.DataFrame({
                'feature1': [1, 2, 3, 4],
                'feature2': [5, 6, 7, 8],
                'label': [0, 1, 0, 1]
            })
            df.to_csv(csv_path, index=False)

            with patch.object(sys, 'argv', ['neuros', 'train', '--csv', str(csv_path)]):
                main()

            # Verify model was saved
            model_path = csv_path.with_suffix('.model.pkl')
            assert model_path.exists()
            assert mock_print.called

    @patch('neuros.models.ModelRegistry')
    @patch('builtins.print')
    def test_save_model_command(self, mock_print, mock_registry_class):
        """Test 'save-model' command."""
        from neuros.models.simple_classifier import SimpleClassifier

        mock_registry = MagicMock()
        mock_metadata = MagicMock()
        mock_metadata.name = 'test_model'
        mock_metadata.version = 'v1.0'
        mock_metadata.model_type = 'SimpleClassifier'
        mock_metadata.file_path = '/path/to/model.pkl'
        mock_registry.save.return_value = mock_metadata
        mock_registry_class.return_value = mock_registry

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            # Write a real pickled model
            import pickle
            model = SimpleClassifier()
            pickle.dump(model, tmp)
            tmp_name = tmp.name

        try:
            with patch.object(sys, 'argv', [
                'neuros', 'save-model',
                '--model-file', tmp_name,
                '--name', 'test_model',
                '--version', 'v1.0'
            ]):
                main()

            # Verify registry.save was called
            assert mock_registry.save.called
        finally:
            Path(tmp_name).unlink()

    @patch('neuros.models.ModelRegistry')
    @patch('builtins.print')
    def test_load_model_command(self, mock_print, mock_registry_class):
        """Test 'load-model' command."""
        mock_registry = MagicMock()
        mock_metadata = MagicMock()
        mock_metadata.name = 'test_model'
        mock_metadata.version = 'v1.0'
        mock_metadata.model_type = 'SimpleClassifier'
        mock_metadata.created_at = '2025-01-01T00:00:00'
        mock_metadata.metrics = {'accuracy': 0.85}

        mock_registry.load.return_value = MagicMock()
        mock_registry.get_metadata.return_value = mock_metadata
        mock_registry.get_latest.return_value = mock_metadata
        mock_registry_class.return_value = mock_registry

        with patch.object(sys, 'argv', ['neuros', 'load-model', '--name', 'test_model']):
            main()

        # Verify registry.load was called
        assert mock_registry.load.called
        assert mock_print.called

    @patch('neuros.models.ModelRegistry')
    @patch('builtins.print')
    def test_list_models_command_table(self, mock_print, mock_registry_class):
        """Test 'list-models' command with table output."""
        mock_registry = MagicMock()
        mock_model1 = MagicMock()
        mock_model1.name = 'model1'
        mock_model1.version = 'v1'
        mock_model1.model_type = 'SVM'
        mock_model1.created_at = '2025-01-01T00:00:00'
        mock_model1.metrics = {'accuracy': 0.85}
        mock_model1.tags = ['eeg']

        mock_registry.list_models.return_value = [mock_model1]
        mock_registry_class.return_value = mock_registry

        with patch.object(sys, 'argv', ['neuros', 'list-models']):
            main()

        assert mock_registry.list_models.called
        assert mock_print.called

    @patch('neuros.models.ModelRegistry')
    @patch('builtins.print')
    def test_list_models_command_json(self, mock_print, mock_registry_class):
        """Test 'list-models' command with JSON output."""
        mock_registry = MagicMock()
        mock_model1 = MagicMock()
        mock_model1.to_dict.return_value = {'name': 'model1', 'version': 'v1'}

        mock_registry.list_models.return_value = [mock_model1]
        mock_registry_class.return_value = mock_registry

        with patch.object(sys, 'argv', ['neuros', 'list-models', '--format', 'json']):
            main()

        assert mock_registry.list_models.called

    @patch('subprocess.run')
    @patch('importlib.util.find_spec')
    def test_dashboard_command(self, mock_find_spec, mock_subprocess):
        """Test 'dashboard' command."""
        # Mock streamlit being available
        mock_find_spec.return_value = MagicMock()

        with patch.object(sys, 'argv', ['neuros', 'dashboard']):
            main()

        # Verify streamlit was launched
        assert mock_subprocess.called
        call_args = mock_subprocess.call_args[0][0]
        assert 'streamlit' in call_args

    @patch('neuros.agents.notebook_agent.NotebookAgent')
    @patch('builtins.print')
    def test_demo_command(self, mock_print, mock_notebook_class):
        """Test 'demo' command."""
        mock_notebook = MagicMock()
        mock_notebook.generate_demo.return_value = '/path/to/demo.ipynb'
        mock_notebook_class.return_value = mock_notebook

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(sys, 'argv', [
                'neuros', 'demo',
                '--task', '2-class motor imagery',
                '--output-dir', tmpdir
            ]):
                main()

            # Verify notebook was generated
            assert mock_notebook.generate_demo.called
            mock_notebook.generate_demo.assert_called_with('2-class motor imagery', duration=3.0)

    @patch('neuros.agents.modality_manager_agent.ModalityManagerAgent')
    @patch('neuros.cli.asyncio.run')
    @patch('builtins.print')
    def test_run_tasks_command(self, mock_print, mock_asyncio_run, mock_manager_class):
        """Test 'run-tasks' command."""
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager

        # Mock the async run_all to return metrics
        mock_asyncio_run.return_value = [
            {'task': 'motor imagery', 'accuracy': 0.8},
            {'task': 'emotion', 'accuracy': 0.75}
        ]

        with patch.object(sys, 'argv', [
            'neuros', 'run-tasks',
            '--tasks', 'motor imagery', 'emotion'
        ]):
            main()

        # Verify run_all was called via asyncio.run
        assert mock_asyncio_run.called

    def test_serve_command(self):
        """Test 'serve' command."""
        with patch('uvicorn.run') as mock_uvicorn:
            with patch.object(sys, 'argv', ['neuros', 'serve', '--host', '0.0.0.0', '--port', '8080']):
                main()

            # Verify uvicorn was called
            assert mock_uvicorn.called
            call_kwargs = mock_uvicorn.call_args[1]
            assert call_kwargs['host'] == '0.0.0.0'
            assert call_kwargs['port'] == 8080

    @patch('neuros.cli.asyncio.run')
    @patch('builtins.print')
    def test_constellation_command(self, mock_print, mock_asyncio_run):
        """Test 'constellation' command."""
        mock_asyncio_run.return_value = None

        with patch.object(sys, 'argv', ['neuros', 'constellation', '--duration', '5.0']):
            main()

        # Verify constellation demo was executed
        assert mock_asyncio_run.called


class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_no_command_fails(self):
        """Test that no command raises SystemExit."""
        with patch.object(sys, 'argv', ['neuros']):
            with pytest.raises(SystemExit):
                _parse_args()

    def test_train_without_csv_fails(self):
        """Test that 'train' without --csv raises SystemExit."""
        with patch.object(sys, 'argv', ['neuros', 'train']):
            with pytest.raises(SystemExit):
                _parse_args()

    def test_save_model_without_required_args_fails(self):
        """Test that 'save-model' without required args raises SystemExit."""
        with patch.object(sys, 'argv', ['neuros', 'save-model', '--model-file', 'model.pkl']):
            with pytest.raises(SystemExit):
                _parse_args()

    def test_load_model_without_name_fails(self):
        """Test that 'load-model' without --name raises SystemExit."""
        with patch.object(sys, 'argv', ['neuros', 'load-model']):
            with pytest.raises(SystemExit):
                _parse_args()

    def test_demo_without_task_fails(self):
        """Test that 'demo' without --task raises SystemExit."""
        with patch.object(sys, 'argv', ['neuros', 'demo']):
            with pytest.raises(SystemExit):
                _parse_args()

    def test_run_tasks_without_tasks_fails(self):
        """Test that 'run-tasks' without --tasks raises SystemExit."""
        with patch.object(sys, 'argv', ['neuros', 'run-tasks']):
            with pytest.raises(SystemExit):
                _parse_args()
