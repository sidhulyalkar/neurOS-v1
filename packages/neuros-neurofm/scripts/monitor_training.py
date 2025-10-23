"""
Training Monitor for NeuroFM-X
Provides real-time analysis of training progress, model health, and debugging tools.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, Optional
import seaborn as sns

class TrainingMonitor:
    """Monitor training progress and model health."""

    def __init__(self, checkpoint_dir: Path = Path("./checkpoints_neurofmx_full_run")):
        self.checkpoint_dir = Path(checkpoint_dir)
        sns.set_style("whitegrid")

    def analyze_checkpoint(self, ckpt_name: str = "latest.pt") -> Dict:
        """Comprehensive checkpoint analysis."""
        ckpt_path = self.checkpoint_dir / ckpt_name

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"\n{'='*80}")
        print(f"Analyzing Checkpoint: {ckpt_name}")
        print(f"{'='*80}\n")

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location='cpu')

        # Extract info
        epoch = ckpt.get('epoch', 'N/A')
        global_step = ckpt.get('global_step', 'N/A')
        best_val_loss = ckpt.get('best_val_loss', 'N/A')

        print(f"📊 Training Progress:")
        print(f"   Epoch: {epoch}")
        print(f"   Global Step: {global_step}")
        print(f"   Best Val Loss: {best_val_loss:.6f}" if isinstance(best_val_loss, float) else f"   Best Val Loss: {best_val_loss}")
        print()

        # Analyze model state
        model_state = ckpt['model_state_dict']
        self._check_model_health(model_state)

        # Compute parameter statistics
        self._compute_param_stats(model_state)

        # Check for training issues
        self._check_training_issues(model_state)

        return {
            'epoch': epoch,
            'global_step': global_step,
            'best_val_loss': best_val_loss,
            'model_health': self._get_health_summary(model_state)
        }

    def _check_model_health(self, model_state: Dict):
        """Check for NaN/Inf and other issues."""
        print(f"🏥 Model Health Check:")

        nan_params = []
        inf_params = []
        zero_params = []

        for name, param in model_state.items():
            if torch.isnan(param).any():
                nan_params.append(name)
            if torch.isinf(param).any():
                inf_params.append(name)
            if (param == 0).all():
                zero_params.append(name)

        if nan_params:
            print(f"   ⚠️  NaN detected in {len(nan_params)} parameters:")
            for name in nan_params[:5]:  # Show first 5
                print(f"      - {name}")
        else:
            print(f"   ✓ No NaN values detected")

        if inf_params:
            print(f"   ⚠️  Inf detected in {len(inf_params)} parameters:")
            for name in inf_params[:5]:
                print(f"      - {name}")
        else:
            print(f"   ✓ No Inf values detected")

        if zero_params:
            print(f"   ⚠️  All-zero parameters: {len(zero_params)}")
            for name in zero_params[:3]:
                print(f"      - {name}")

        print()

    def _compute_param_stats(self, model_state: Dict):
        """Compute statistics for model parameters."""
        print(f"📈 Parameter Statistics:")

        # Group parameters by layer type
        stats = {}
        for name, param in model_state.items():
            layer_type = self._get_layer_type(name)

            if layer_type not in stats:
                stats[layer_type] = {
                    'mean': [],
                    'std': [],
                    'min': [],
                    'max': [],
                    'count': 0
                }

            stats[layer_type]['mean'].append(param.mean().item())
            stats[layer_type]['std'].append(param.std().item())
            stats[layer_type]['min'].append(param.min().item())
            stats[layer_type]['max'].append(param.max().item())
            stats[layer_type]['count'] += param.numel()

        # Print summary
        for layer_type, layer_stats in sorted(stats.items()):
            mean_avg = np.mean(layer_stats['mean'])
            std_avg = np.mean(layer_stats['std'])
            total_params = layer_stats['count']

            print(f"   {layer_type:20s}: mean={mean_avg:7.4f}, std={std_avg:7.4f}, params={total_params:>10,}")

        print()

    def _get_layer_type(self, param_name: str) -> str:
        """Extract layer type from parameter name."""
        if 'tokenizer' in param_name:
            return 'tokenizer'
        elif 'backbone' in param_name:
            return 'backbone'
        elif 'fusion' in param_name:
            return 'fusion'
        elif 'popt' in param_name:
            return 'popt'
        elif 'heads' in param_name:
            if 'encoder' in param_name:
                return 'head_encoder'
            elif 'decoder' in param_name:
                return 'head_decoder'
            elif 'contrastive' in param_name:
                return 'head_contrastive'
            else:
                return 'heads'
        else:
            return 'other'

    def _check_training_issues(self, model_state: Dict):
        """Check for common training issues."""
        print(f"🔍 Training Issue Detection:")

        issues_found = False

        # Check for very small/large weights
        for name, param in model_state.items():
            if param.abs().max() > 100:
                print(f"   ⚠️  Large weights in {name}: max={param.abs().max():.2f}")
                issues_found = True
            if param.abs().mean() < 1e-6 and 'bias' not in name:
                print(f"   ⚠️  Very small weights in {name}: mean={param.abs().mean():.2e}")
                issues_found = True

        if not issues_found:
            print(f"   ✓ No obvious training issues detected")

        print()

    def _get_health_summary(self, model_state: Dict) -> str:
        """Get overall model health summary."""
        nan_count = sum(1 for p in model_state.values() if torch.isnan(p).any())
        inf_count = sum(1 for p in model_state.values() if torch.isinf(p).any())

        if nan_count > 0 or inf_count > 0:
            return "UNHEALTHY"
        else:
            return "HEALTHY"

    def plot_training_progress(self, log_file: Optional[Path] = None):
        """Plot training progress from logs."""
        # TODO: Implement once logging is added to training script
        pass

    def compare_checkpoints(self, ckpt1: str, ckpt2: str):
        """Compare two checkpoints to track parameter evolution."""
        print(f"\n{'='*80}")
        print(f"Comparing Checkpoints: {ckpt1} vs {ckpt2}")
        print(f"{'='*80}\n")

        state1 = torch.load(self.checkpoint_dir / ckpt1, map_location='cpu')['model_state_dict']
        state2 = torch.load(self.checkpoint_dir / ckpt2, map_location='cpu')['model_state_dict']

        print(f"Parameter Changes:")

        for name in sorted(state1.keys()):
            if name in state2:
                diff = (state2[name] - state1[name]).abs().mean().item()
                print(f"   {name:60s}: Δ={diff:.6f}")

        print()

    def export_analysis_report(self, output_path: Path = Path("./training_analysis.txt")):
        """Export detailed analysis report."""
        with open(output_path, 'w') as f:
            # Redirect prints to file
            import sys
            original_stdout = sys.stdout
            sys.stdout = f

            self.analyze_checkpoint("latest.pt")

            sys.stdout = original_stdout

        print(f"✓ Analysis report saved to: {output_path}")


def main():
    """Run training monitor."""
    import argparse

    parser = argparse.ArgumentParser(description="Monitor NeuroFM-X training")
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_neurofmx_full_run',
                        help='Path to checkpoint directory')
    parser.add_argument('--checkpoint', type=str, default='latest.pt',
                        help='Checkpoint file to analyze')
    parser.add_argument('--compare', type=str, nargs=2, metavar=('CKPT1', 'CKPT2'),
                        help='Compare two checkpoints')
    parser.add_argument('--export', action='store_true',
                        help='Export analysis report to file')

    args = parser.parse_args()

    monitor = TrainingMonitor(checkpoint_dir=Path(args.checkpoint_dir))

    if args.compare:
        monitor.compare_checkpoints(args.compare[0], args.compare[1])
    else:
        result = monitor.analyze_checkpoint(args.checkpoint)

        if args.export:
            monitor.export_analysis_report()

    print(f"\n{'='*80}")
    print(f"Monitoring Complete")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
