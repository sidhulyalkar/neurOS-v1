# Energy Flow and Information Landscape Analysis Module

## Overview

The `energy_flow.py` module implements advanced analysis tools for understanding information processing and energy dynamics in NeuroFMX neural networks. This module is based on:

- **Tishby & Zaslavsky (2015)**: Deep learning and the information bottleneck principle
- **Schwartz-Ziv & Tishby (2017)**: Opening the black box of deep neural networks
- **Seifert (2012)**: Stochastic thermodynamics, fluctuation theorems and molecular machines

## Module Location

```
packages/neuros-neurofm/src/neuros_neurofm/interpretability/energy_flow.py
```

## Components

### 1. Data Structures

#### `MutualInformationEstimate`
Results from mutual information estimation.

```python
@dataclass
class MutualInformationEstimate:
    I_XZ: float                              # I(X;Z) - input-latent MI
    I_ZY: float                              # I(Z;Y) - latent-output MI
    method: str                              # Estimation method used
    confidence_interval: Optional[Tuple]     # Bootstrap CI if computed
    metadata: Optional[Dict]                 # Additional information
```

#### `InformationPlane`
Tishby's information plane data structure.

```python
@dataclass
class InformationPlane:
    layers: List[str]                        # Layer names
    I_XZ_per_layer: np.ndarray              # I(X;Z) for each layer
    I_ZY_per_layer: np.ndarray              # I(Z;Y) for each layer
    epochs: Optional[List[int]]             # Training epochs (if temporal)
    I_XZ_trajectory: Optional[np.ndarray]   # MI trajectory over time
    I_ZY_trajectory: Optional[np.ndarray]   # MI trajectory over time
```

#### `EnergyFunction`
Energy landscape representation.

```python
@dataclass
class EnergyFunction:
    grid: np.ndarray                         # Grid coordinates
    energy: np.ndarray                       # Energy values at grid points
    latent_dim: int                          # Latent space dimension
    method: str                              # Estimation method
    pca_basis: Optional[np.ndarray]         # PCA basis for >2D projections
```

#### `Basin`
Energy basin (stable state).

```python
@dataclass
class Basin:
    centroid: np.ndarray                     # Basin center
    energy: float                            # Energy at centroid
    volume: float                            # Basin volume estimate
    samples: np.ndarray                      # Points in basin
    stability: float                         # Local curvature measure
```

#### `EntropyProductionEstimate`
Entropy production along trajectories.

```python
@dataclass
class EntropyProductionEstimate:
    entropy_production_rate: np.ndarray      # dS/dt per timestep
    dissipation_rate: float                  # Total energy dissipation
    nonequilibrium_score: float              # Distance from equilibrium
    trajectories: np.ndarray                 # Trajectory data
```

---

### 2. MINENetwork (Mutual Information Neural Estimation)

Small neural network for estimating mutual information using the Donsker-Varadhan representation.

```python
class MINENetwork(nn.Module):
    def __init__(
        self,
        x_dim: int,          # Dimension of X
        z_dim: int,          # Dimension of Z
        hidden_dim: int = 128,
        n_layers: int = 3
    )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute statistics network T(x, z)"""
```

**Example:**
```python
mine_net = MINENetwork(x_dim=10, z_dim=8, hidden_dim=128, n_layers=3)
x = torch.randn(100, 10)
z = torch.randn(100, 8)
T_xz = mine_net(x, z)  # Output: (100, 1)
```

---

### 3. InformationFlowAnalyzer

Analyzes information flow through neural network layers.

#### Methods

##### `estimate_mutual_information()`
Compute I(X;Z) and I(Z;Y) for layers.

```python
def estimate_mutual_information(
    self,
    X: torch.Tensor,                    # Input data (batch, input_dim)
    Z_layers: Union[torch.Tensor, List[torch.Tensor]],  # Layer activations
    Y: Optional[torch.Tensor] = None,   # Output/target (batch, output_dim)
    method: str = 'mine',               # 'mine', 'knn', or 'histogram'
    n_bootstrap: int = 0                # Bootstrap samples for CI
) -> Dict[str, MutualInformationEstimate]:
```

**Methods:**
- **`mine`**: MINE (Mutual Information Neural Estimation) - most accurate, slower
- **`knn`**: k-NN estimator (Kraskov method) - good balance
- **`histogram`**: Binning-based - fastest, less accurate in high dimensions

**Example:**
```python
from neuros_neurofm.interpretability.energy_flow import InformationFlowAnalyzer

analyzer = InformationFlowAnalyzer(device='cuda')

# Sample data
X = torch.randn(1000, 20)  # Input
Z = torch.randn(1000, 10)  # Latent
Y = torch.randn(1000, 5)   # Output

# Estimate MI using k-NN method
results = analyzer.estimate_mutual_information(
    X, [Z], Y,
    method='knn',
    n_bootstrap=100  # Compute 95% CI
)

for layer_name, result in results.items():
    print(f"{layer_name}:")
    print(f"  I(X;Z) = {result.I_XZ:.4f}")
    print(f"  I(Z;Y) = {result.I_ZY:.4f}")
```

##### `information_plane()`
Compute Tishby's information plane: I(X;T) vs I(T;Y).

```python
def information_plane(
    self,
    activations: Dict[str, torch.Tensor],  # Layer activations
    X: torch.Tensor,                       # Input data
    Y: torch.Tensor,                       # Output data
    method: str = 'mine'
) -> InformationPlane:
```

**Example:**
```python
# Collect activations from model
activations = {
    'encoder_0': torch.randn(1000, 64),
    'encoder_1': torch.randn(1000, 128),
    'encoder_2': torch.randn(1000, 64),
    'decoder_0': torch.randn(1000, 32),
}

info_plane = analyzer.information_plane(activations, X, Y, method='knn')

# Visualize
fig = analyzer.visualize_information_plane(
    info_plane,
    save_path='info_plane.png',
    show=True
)
```

##### `information_bottleneck_curve()`
Compute information bottleneck tradeoff curve.

```python
def information_bottleneck_curve(
    self,
    info_plane: InformationPlane,
    beta_range: np.ndarray = np.logspace(-2, 2, 20)
) -> Tuple[np.ndarray, np.ndarray]:
```

The IB objective is: `min_Z I(X;Z) - β*I(Z;Y)`

**Example:**
```python
I_XZ_curve, I_ZY_curve = analyzer.information_bottleneck_curve(
    info_plane,
    beta_range=np.logspace(-2, 2, 20)
)

# Plot tradeoff curve
import matplotlib.pyplot as plt
plt.plot(I_XZ_curve, I_ZY_curve, 'o-')
plt.xlabel('I(X;Z)')
plt.ylabel('I(Z;Y)')
plt.title('Information Bottleneck Tradeoff')
plt.show()
```

---

### 4. EnergyLandscape

Estimates energy landscape of latent space where `p(z) ∝ exp(-U(z))`.

#### Methods

##### `estimate_landscape()`
Approximate energy function U(z).

```python
def estimate_landscape(
    self,
    latents: torch.Tensor,          # Latent samples (n_samples, latent_dim)
    method: str = 'density',        # 'score', 'quadratic', or 'density'
    grid_resolution: int = 50,
    n_components_2d: int = 2
) -> EnergyFunction:
```

**Methods:**
- **`density`**: U(z) = -log p(z) via GMM density estimation (recommended)
- **`score`**: Estimate ∇U(z) via score matching
- **`quadratic`**: Local quadratic approximation (Gaussian assumption)

**Example:**
```python
from neuros_neurofm.interpretability.energy_flow import EnergyLandscape

landscape_analyzer = EnergyLandscape(device='cuda')

# Generate latents with multiple modes
latents = torch.cat([
    torch.randn(300, 2) + torch.tensor([2.0, 2.0]),
    torch.randn(300, 2) + torch.tensor([-2.0, -2.0]),
    torch.randn(400, 2)
], dim=0)

landscape = landscape_analyzer.estimate_landscape(
    latents,
    method='density',
    grid_resolution=100
)

print(f"Energy range: [{landscape.energy.min():.3f}, {landscape.energy.max():.3f}]")
```

##### `find_basins()`
Identify energy basins (stable states).

```python
def find_basins(
    self,
    landscape: EnergyFunction,
    num_basins: int = 5,
    min_depth: float = 0.5
) -> List[Basin]:
```

**Example:**
```python
basins = landscape_analyzer.find_basins(
    landscape,
    num_basins=3,
    min_depth=0.5
)

for i, basin in enumerate(basins):
    print(f"Basin {i}:")
    print(f"  Centroid: {basin.centroid}")
    print(f"  Energy: {basin.energy:.3f}")
    print(f"  Stability: {basin.stability:.3f}")
```

##### `compute_barriers()`
Compute energy barriers between basins.

```python
def compute_barriers(
    self,
    landscape: EnergyFunction,
    basins: List[Basin]
) -> np.ndarray:  # (n_basins, n_basins) barrier matrix
```

**Example:**
```python
barriers = landscape_analyzer.compute_barriers(landscape, basins)
print("Energy barriers:")
print(barriers)
```

##### `visualize_landscape_2d()`
Visualize 2D energy landscape.

```python
def visualize_landscape_2d(
    self,
    landscape: EnergyFunction,
    basins: Optional[List[Basin]] = None,
    latents: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
```

**Example:**
```python
fig = landscape_analyzer.visualize_landscape_2d(
    landscape,
    basins=basins,
    latents=latents,
    save_path='energy_landscape.png',
    show=True
)
```

---

### 5. EntropyProduction

Measures entropy production along trajectories.

#### Methods

##### `estimate_entropy_production()`
Estimate dS/dt along trajectories.

```python
def estimate_entropy_production(
    self,
    trajectories: torch.Tensor,  # (n_trials, n_timesteps, n_dims)
    dt: float = 0.01
) -> EntropyProductionEstimate:
```

Uses the approximation: `dS/dt ≈ ||dx/dt||² / (2*D)` where D is the diffusion coefficient.

**Example:**
```python
from neuros_neurofm.interpretability.energy_flow import EntropyProduction

entropy_analyzer = EntropyProduction(device='cuda')

# Generate trajectories
n_trials = 50
n_timesteps = 100
n_dims = 5
trajectories = torch.cumsum(
    torch.randn(n_trials, n_timesteps, n_dims) * 0.1,
    dim=1
)

entropy_prod = entropy_analyzer.estimate_entropy_production(
    trajectories,
    dt=0.01
)

print(f"Dissipation rate: {entropy_prod.dissipation_rate:.4f}")
print(f"Nonequilibrium score: {entropy_prod.nonequilibrium_score:.4f}")
```

##### `dissipation_rate()`
Get total energy dissipation rate.

```python
def dissipation_rate(
    self,
    entropy_production: EntropyProductionEstimate
) -> float:
```

##### `nonequilibrium_score()`
Get distance from equilibrium.

```python
def nonequilibrium_score(
    self,
    entropy_production: EntropyProductionEstimate
) -> float:
```

##### `visualize_entropy_production()`
Visualize entropy production over time.

```python
def visualize_entropy_production(
    self,
    entropy_production: EntropyProductionEstimate,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
```

**Example:**
```python
fig = entropy_analyzer.visualize_entropy_production(
    entropy_prod,
    save_path='entropy_production.png',
    show=True
)
```

---

### 6. Utility Functions

#### `compute_information_plane_trajectory()`
Compute information plane trajectory over training epochs.

```python
def compute_information_plane_trajectory(
    model: nn.Module,
    checkpoints: List[str],              # Checkpoint paths
    data_loader,                         # DataLoader
    layer_names: List[str],              # Layer names to analyze
    device: str = "cuda",
    method: str = 'knn'
) -> InformationPlane:
```

**Example:**
```python
from neuros_neurofm.interpretability.energy_flow import compute_information_plane_trajectory

checkpoints = [
    'checkpoint_epoch_0.pt',
    'checkpoint_epoch_10.pt',
    'checkpoint_epoch_20.pt',
    'checkpoint_epoch_30.pt',
]

layer_names = ['encoder.0', 'encoder.1', 'encoder.2']

info_plane_trajectory = compute_information_plane_trajectory(
    model=my_model,
    checkpoints=checkpoints,
    data_loader=val_loader,
    layer_names=layer_names,
    method='knn'
)

# Visualize temporal evolution
for epoch_idx in range(len(checkpoints)):
    I_XZ = info_plane_trajectory.I_XZ_trajectory[epoch_idx]
    I_ZY = info_plane_trajectory.I_ZY_trajectory[epoch_idx]
    plt.plot(I_XZ, I_ZY, 'o-', label=f'Epoch {epoch_idx}')
plt.xlabel('I(X;Z)')
plt.ylabel('I(Z;Y)')
plt.legend()
plt.show()
```

---

## Complete Usage Example

```python
import torch
from neuros_neurofm.interpretability.energy_flow import (
    InformationFlowAnalyzer,
    EnergyLandscape,
    EntropyProduction
)

# 1. Information Flow Analysis
print("=== INFORMATION FLOW ANALYSIS ===")
analyzer = InformationFlowAnalyzer(device='cuda', verbose=True)

X = torch.randn(1000, 20)  # Input
Z = torch.randn(1000, 10)  # Latent
Y = torch.randn(1000, 5)   # Output

# Estimate mutual information
mi_results = analyzer.estimate_mutual_information(
    X, [Z], Y, method='knn'
)

# Compute information plane
activations = {
    'layer_0': torch.randn(1000, 64),
    'layer_1': torch.randn(1000, 128),
    'layer_2': torch.randn(1000, 64),
}
info_plane = analyzer.information_plane(activations, X, Y)
analyzer.visualize_information_plane(info_plane, save_path='info_plane.png')

# 2. Energy Landscape Analysis
print("\n=== ENERGY LANDSCAPE ANALYSIS ===")
landscape_analyzer = EnergyLandscape(device='cuda', verbose=True)

latents = torch.randn(1000, 2)
landscape = landscape_analyzer.estimate_landscape(latents, method='density')
basins = landscape_analyzer.find_basins(landscape, num_basins=3)
barriers = landscape_analyzer.compute_barriers(landscape, basins)

landscape_analyzer.visualize_landscape_2d(
    landscape,
    basins=basins,
    latents=latents,
    save_path='energy_landscape.png'
)

# 3. Entropy Production Analysis
print("\n=== ENTROPY PRODUCTION ANALYSIS ===")
entropy_analyzer = EntropyProduction(device='cuda', verbose=True)

trajectories = torch.cumsum(torch.randn(50, 100, 5) * 0.1, dim=1)
entropy_prod = entropy_analyzer.estimate_entropy_production(trajectories, dt=0.01)

entropy_analyzer.visualize_entropy_production(
    entropy_prod,
    save_path='entropy_production.png'
)

print("\nAnalysis complete!")
```

---

## Integration with NeuroFMX

The module is fully integrated into the NeuroFMX interpretability suite:

```python
from neuros_neurofm.interpretability import (
    InformationFlowAnalyzer,
    EnergyLandscape,
    EntropyProduction,
    InformationPlane,
    MutualInformationEstimate,
)
```

---

## Dependencies

- PyTorch
- NumPy
- SciPy (stats, spatial, optimize, ndimage)
- scikit-learn (neighbors, mixture, decomposition)
- Matplotlib
- Seaborn

---

## References

1. Belghazi et al. (2018) "Mutual Information Neural Estimation" (MINE)
2. Kraskov et al. (2004) "Estimating mutual information" (k-NN estimator)
3. Tishby & Zaslavsky (2015) "Deep learning and the information bottleneck principle"
4. Schwartz-Ziv & Tishby (2017) "Opening the black box of deep neural networks"
5. Seifert (2012) "Stochastic thermodynamics, fluctuation theorems and molecular machines"

---

## Code Statistics

- **Total lines**: 1,275
- **Code lines**: 923
- **Classes**: 9
- **Functions**: 1
- **Docstrings**: 35
