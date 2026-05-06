#!/bin/bash
# neuros-astro installation script
# Run this in your conda environment

set -e  # Exit on error

echo "=========================================="
echo "neuros-astro Installation"
echo "=========================================="
echo ""

# Check if in neurOS-v1 directory
if [[ ! -d "packages/neuros-astro" ]]; then
    echo "❌ Error: Run this from neurOS-v1 root directory"
    echo "   cd /mnt/c/Users/sidso/Documents/neurOS-v1"
    exit 1
fi

# Check Python version
echo "🐍 Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "   Python $python_version"

# Navigate to package
cd packages/neuros-astro
echo ""

# Install package
echo "📦 Installing neuros-astro with all features..."
pip install -e ".[all]"
echo "   ✓ Installation complete!"
echo ""

# Verify CLI
echo "🔧 Verifying CLI..."
if command -v neuros-astro &> /dev/null; then
    echo "   ✓ neuros-astro CLI installed"
else
    echo "   ⚠️  CLI not found in PATH"
fi
echo ""

# Run tests
echo "🧪 Running tests..."
pytest tests/ -v --tb=short | tail -20
echo ""

# Quick import test
echo "🔍 Testing imports..."
python -c "
from neuros_astro import __version__
from neuros_astro.visualization import plot_event_raster
from neuros_astro.io.synthetic import generate_synthetic_astro_traces
from neuros_astro.events.event_detection import detect_events_from_traces
print('   ✓ All imports successful!')
print(f'   Version: {__version__}')
"
echo ""

echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run quick demo:"
echo "     python examples/05_get_started_today.py"
echo ""
echo "  2. Process Allen data:"
echo "     python examples/06_process_allen_data.py"
echo ""
echo "  3. Open Jupyter notebooks:"
echo "     cd notebooks && jupyter notebook"
echo ""
echo "  4. Check CLI:"
echo "     neuros-astro --help"
echo ""
