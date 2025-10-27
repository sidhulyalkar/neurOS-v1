Why fractals belong in neuroFMx (quick evidence)

Neurons & dendrites are fractal, and fractal dimension (FD) reflects wiring/energy trade-offs. Dendritic arbors exhibit an “effective” fractal dimension measured via 3D box-counting; manipulating branching/weaving changes D and systematically alters predicted inter-neuron connectivity and cost—suggesting neurons optimize D under constraints. 
Nature

Cortical folding is fractal across scales and FD is a robust biomarker. FD tracks development/aging and neurodegeneration and can outperform thickness/gyrification for clinical differentiation (e.g., AD). Methodology matters (scale-interval selection, ribbons vs. surfaces). 
Nature
+2
ADNI
+2

Networks and time series show scale-free structure tied to state of consciousness. Functional connectivity graphs and temporal activity (Higuchi FD/1-f) decrease with reduced consciousness, linking fractality to brain state. 
PLOS

Comprehensive reviews/book chapters: multi-level evidence (anatomy→physiology→behavior) that fractality is ubiquitous in brain structure and dynamics. 
PMC
+2
cmadan.com
+2

Takeaway: fractal features aren’t “nice to have”—they are mechanistic handles on morphology, network topology, and temporal statistics. Let’s make them first-class citizens in neuroFMx.

neuroFMx additions (modules you can drop into the repo)
1) mechanistic/fractals/metrics.py — fast, batched estimators

Add GPU-friendly estimators with windowed batching so they run during training/eval.

Temporal (per channel / per unit / per latent):

Higuchi FD (HF) & Katz FD

Detrended Fluctuation Analysis exponent α (DFA)

Spectral slope β in 1/f^β band (log–log PSD regression, excluding oscillatory peaks)

Hurst exponent H (via DFA or R/S)

Graph/Population:

Graph fractal dimension of functional connectivity (box-covering on thresholded graphs, multi-ε regression)

Degree/weight distributions power-law fit diagnostics (Clauset–Shalizi style)

Multifractal spectrum (partition function τ(q), Legendre to f(α)) for population rates

Anatomy (offline preprocess):

Cortical FD from MRI ribbon/surface (3D box-counting; pipeline replicating robust scale-interval choice). Store as subject/session covariates. 
Nature
+1

Why: these metrics let you track scale-free structure across layers, heads, time, tasks, and subjects; they become targets for priors, covariance analysis, and mixed-effects models. (Evidence base for each metric: dendrite FD & wiring/efficiency, cortical FD & disease/aging, network FD & consciousness. 
Nature
+2
ADNI
+2
)

2) mechanistic/fractals/probes.py — plug-and-play interpretability probes

Latent-space FD tracker: compute HF/DFA/H on Perceiver latents, PopT tokens, and Mamba hidden states over training; log in TensorBoard/W&B.

Attention–fractal coupling: correlate attention weights (Perceiver cross-attn, PopT self-attn) with local FD/H changes in inputs/latents over trials/conditions.

Causal ablation by scale: zero out (or noise-inject) only the high-frequency fractal components vs. low-frequency ones; measure impact on decoding and on β/H in downstream latents.

Connectivity FD probe: build per-batch FC graphs from latent channels (correlations or OLS partials), compute graph FD, and relate to behavior/state labels. 
PLOS

Structure→Function mapping: regress decoding performance against subject cortical FD/tissue FD covariates; test if individuals with lower cortical FD need stronger latent scale-mixture (biologically plausible link). 
PMC
+1

3) Fractal priors/regularizers (drop into the training objectives)

Add these as optional loss terms with schedule/weighting in config.yaml.

1/f spectral prior (temporal): Penalize deviation of latent/feature PSD from target slope β* band (e.g., 0.5–2 Hz-decade) while not suppressing oscillatory peaks.
L_spec = || (log P(f) - (c - β* log f))_mask ||^2
Rationale: real neural activity shows 1/f-like spectra; regularizing latents toward realistic scale mixtures yields more brain-like—and interpretable—representations. 
PLOS

Multifractal moment smoothness: Encourage smooth τ(q) curve across q (prevents degenerate single-scale solutions) and match empirical τ(q) from data segments.

Graph fractality prior: For latent FC graphs, penalize deviation from target fractal dimension D* estimated from the subject/session condition (e.g., sleep vs. awake); or just maximize identifiability by stabilizing D across seeds.

Dale-aware scale prior (optional): Combine with your E/I constraints (if/when added) so inhibitory subspaces preferentially stabilize high-β (steeper) components, echoing known spectral fingerprints.

4) Fractalized backbones & encodings (minimal invasive, high gain)

Mamba-SSM with fractional kernels: swap some convolutional/state-mixing kernels for power-law impulse responses (approximate fractional derivative/integrator via truncated power-law taps). This bakes long-memory dynamics directly into the SSM. (Bridges to fractional-order neural dynamics models emerging in the literature.)

Fractal positional encodings: replace/add to sinusoidal encodings with log-spaced, multi-rate encodings (geometric scale steps), or a wavelet scattering front-end feeding Perceiver, which is provably stable to deformations and captures multi-scale self-similarity.

PopT scale-mixture tokens: for each neuron/time, add a tiny scale pyramid (e.g., wavelet energies at octave scales) to the token’s features so aggregation “sees” the scale context explicitly.

5) Fractal stimulus & environment generators (for causal in-silico experiments)

Add mechanistic/fractals/stimuli.py:

1/f^β textures & fractal Brownian motion (fBm) time series generators (β tunable, colored noise + optional oscillatory peaks).

Random midpoint displacement / diamond-square for spatial fractal surfaces (for visual paradigms or synthetic “neural field” targets).

Multiplicative cascade generators (e.g., Mandelbrot cascades) for multifractal inputs.

Use them to:

Probe units & heads: find β (and multifractal spectrum) that maximally drives a unit/decoder; map “scale tuning curves” akin to orientation tuning.

Train diffusion with fractal priors: when doing latent diffusion, add a discriminator/energy that nudges generated latents toward realistic β/H/τ(q) statistics; this keeps generative rollouts neuro-plausible.

Decode imagery with fractal constraints: if/when coupling to vision models, incorporate fractal priors à la brain→image decoding work that steers generative models using brain signals. 
PLOS

6) Differentiable fractal simulators (mechanistic “digital twins”)

Add mechanistic/fractals/simulators.py:

Fractional Ornstein–Uhlenbeck (fOU) modules (reparam in frequency domain for backprop) to simulate latent neural drives with long memory (H≈0.7–0.9).

Dendrite-inspired branching growth: simple parametric arbor growth with target effective FD D (from the 3D box-counting paper); link D to a cost term (wire length/metabolic proxy) and a connectivity payoff to reproduce the D-optimized trade-off seen empirically. Train small controllers that pick D per-unit/region. 
Nature

Use cases:

Fit simulator params to a subject’s data (learn their “fractal fingerprint”); compare across conditions (sleep, task, disease).

Feed simulator latents into neuroFMx backbone to test if fractal priors cause better decoding/forecasting and more stable attention patterns.

Concrete Code Hooks:
packages/neuros-neurofm/
  mechanistic/fractals/
    __init__.py
    metrics.py           # hf_fd, dfa_alpha, hurst, spectral_beta, graph_fd, multifractal_spectrum
    probes.py            # latent_fd_probe, attention_fd_coupling, causal_scale_ablation, fc_fd_probe
    stimuli.py           # fbm_series(beta), one_over_f_texture(beta), cascade_multifractal(...)
    simulators.py        # fractional_OU(H), dendrite_growth(target_D, cost_coeffs)
    regularizers.py      # l_spec_1overf(...), l_multifractal(...), l_graph_fd(...)


Trainer integration (train.py or LightningModule):
# forward()
latents = model(inputs)

# probes
fd_stats = fractals.metrics.batch_higuchi(latents)   # [B, L]
beta = fractals.metrics.spectral_beta(latents)
graphD = fractals.metrics.graph_fractal_dim(latents)

# losses
loss += λ1 * fractals.regularizers.l_spec_1overf(latents, beta_target)
loss += λ2 * fractals.regularizers.l_multifractal(latents, tau_target)
loss += λ3 * fractals.regularizers.l_graph_fd(latents, D_target)

# logging
log_dict({"fd_mean": fd_stats.mean(), "beta": beta.mean(), "graphFD": graphD.mean()})

Config (configs/neurofm_fractals.yaml):

fractals:
  enable: true
  priors:
    spectral_beta_target: 1.0
    lambda_spec: 0.05
    lambda_mf: 0.02
    lambda_graphfd: 0.01
  probes:
    attention_fd_coupling: true
    causal_scale_ablation:
      mode: "high_beta_only"
      ablate_ratio: 0.15
  stimuli:
    fbm_beta_grid: [0.5, 1.0, 1.5]


Validation & science workflows you unlock

Does latent fractality track brain state?
Run decoding across conditions; check if latent β/H/graph-FD increase with alertness and decrease with anesthesia/sleep—as reported for functional connectivity/time-series. 
PLOS

Are dendrite-like scale priors beneficial?
Enable fractional kernels in Mamba; expect improved long-horizon forecasting and smoother τ(q); compare to vanilla kernels.

Individual differences & prognosis:
Regress subject cortical FD (MRI) against model hyperparameters learned (e.g., best β*); test if lower cortical FD → steeper learned spectral priors or reduced graph FD—consistent with aging/degeneration literature. 


Mechanism discovery via ablation:
If ablation of high-β latent components collapses performance on tasks relying on fine temporal features (e.g., EMG bursts), that’s evidence those scale-specific components are causal.

Reading list (what each guides)

Neurons optimize fractal D for connectivity vs. cost → motivates dendrite-growth simulator & D-aware priors. 
Nature

FD for cortex: methodology & clinical value → informs robust FD pipelines and subject covariates. 
Nature

Scale-free networks/time series & consciousness → motivates graph-FD and β/H probes as state markers. 

Reviews/chapters on fractal brain → global synthesis and pitfalls; good for methods and scale selection. 

How this fits your broader mechanistic toolbox

Works alongside your activation patching/ablation and latent circuit inference: scale-specific perturbations are a new axis of mechanistic testing.

Pairs with diffusion decoding: add an energy term so generated latents/images respect β/H/τ(q), yielding physiology-consistent reconstructions.

Bridges biological plausibility: fractional dynamics and dendrite growth embed real constraints, aligning model internals with interpretable parameters (H, D, β).


neuroFMx – Fractal Geometry Integration Plan (for Claude)

Goal: Add a first-class Fractals Mechanistic Toolbox to packages/neuros-neurofm that (1) computes fast, batched fractal metrics across signals/latents/graphs, (2) injects fractal priors/regularizers into training, (3) provides fractal stimuli & differentiable simulators for causal probing, and (4) logs/visualizes fractal behavior during training and evaluation.

Constraints:

PyTorch-first, GPU-compatible where feasible (vectorized; no Python loops on hot paths).

Clean API, minimal touching of existing modules except for documented hooks.

All features toggleable via config flags.

CI + unit tests + runnable example.

0) Where to put things

packages/neuros-neurofm/
  /src/neuros_neurofm/interpretability/
    fractals/
      __init__.py
      metrics.py           # core batched metric estimators (time, graph, multifractal)
      regularizers.py      # differentiable loss terms using metrics
      probes.py            # callbacks & utilities to analyze/ablate by scale
      stimuli.py           # 1/f^β, fBm, multifractal cascades
      simulators.py        # fractional OU, dendrite-growth toy simulator
      plotting.py          # small helpers for FD/β/H curves & attention coupling
  configs/
    neurofm_fractals.yaml  # default config enabling probes/priors
  trainers/
    callbacks_fractals.py  # Lightning callbacks for logging & intervention
  docs/
    methods_fractals.md    # method write-up; how we compute & why
  tests/
    test_fractals_metrics.py
    test_fractals_regularizers.py
    test_fractals_probes.py
    test_fractals_stimuli.py
    test_fractals_simulators.py

configs/neurofm_fractals.yaml
fractals:
  enable: true

  metrics:
    compute_on:
      - "inputs"      # raw modality streams after tokenization
      - "latents"     # Perceiver latents, PopT tokens, SSM states
    window_size: 1024
    hop_size: 256
    sample_rate_hz: 1000  # for PSD slopes; overridden per modality if provided

    temporal:
      higuchi_kmax: 8
      dfa_orders: [4, 16, 64, 256]
      psd_freq_band: [2.0, 80.0]   # Hz; mask out 0-2Hz drift & >80Hz noise
      exclude_peaks: true          # notch around detected peaks before slope fit
    multifractal:
      q_vals: [-5,-3,-1,0,1,3,5]
      scales: [8,16,32,64,128,256]

    graph:
      corr_threshold: 0.3
      box_eps: [1,2,3,4,5,6]       # graph box-covering eps values

  priors:
    spectral_beta_target: 1.0
    lambda_spec: 0.05
    lambda_mf: 0.02
    lambda_graphfd: 0.01
    warmup_steps: 1000             # start applying priors after warmup

  probes:
    attention_fd_coupling: true
    causal_scale_ablation:
      enable: true
      mode: "high_beta_only"       # ["high_beta_only","low_beta_only"]
      ablate_ratio: 0.15

  backbone:
    fractional_kernels:
      enable: true
      n_taps: 33
      min_alpha: 0.2               # fractional exponent range
      max_alpha: 0.8
    scale_encodings:
      enable: true
      wavelet_scales: [8,16,32,64]

  stimuli:
    enable: true
    fbm_beta_grid: [0.5, 1.0, 1.5]
    n_seconds: 10
    sample_rate_hz: 1000

  log:
    tb_hist_every_n: 500
    wandb: true


metrics.py
# metrics.py (key signatures)

from typing import Tuple, Dict, Optional
import torch

def higuchi_fd(x: torch.Tensor, kmax: int = 8, dim: int = -1) -> torch.Tensor:
    """
    x: (..., T) time-series; returns FD per series shape (...)
    Vectorized Higuchi estimator. No gradients required.
    """

def dfa_alpha(x: torch.Tensor, windows: torch.Tensor) -> torch.Tensor:
    """
    Detrended Fluctuation Analysis exponent α.
    x: (..., T); windows: (W,) integers of window sizes.
    Returns α per series shape (...).
    """

def hurst_from_dfa(alpha: torch.Tensor) -> torch.Tensor:
    """H ≈ α for fGn; convenience mapping."""

def spectral_beta(
    x: torch.Tensor,
    fs: float,
    freq_band: Tuple[float, float],
    exclude_peaks: bool = True
) -> torch.Tensor:
    """
    Log–log slope β of PSD in band. Uses Welch/FFT; masks narrow peaks before regression.
    Returns β per series shape (...).
    """

def multifractal_spectrum(
    x: torch.Tensor,
    q_vals: torch.Tensor,
    scales: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Compute τ(q), α(q), f(α) via partition function approach.
    Returns dict { 'tau': (...., Q), 'alpha': (..., Q), 'falpha': (..., Q) }.
    """

def graph_fractal_dim(
    adj: torch.Tensor,
    eps_values: torch.Tensor
) -> torch.Tensor:
    """
    Box-covering FD on (N,N) adjacency (binary or weighted-thresholded).
    Returns FD scalar per graph (batchable).
    """

def functional_connectivity_latents(
    latents: torch.Tensor,
    method: str = "pearson",
    thresh: float = 0.3
) -> torch.Tensor:
    """
    latents: (B, T, C). Returns adjacency (B, C, C) after threshold.
    """

Expectations/notes

Provide efficient windowing (unfold) and batch FFT.

For exclude_peaks, implement simple peak detection (prominence-based) and mask ±Δf bins.

Unit tests: compare against reference (numpy/scipy) on small synthetic signals.


regularizers.py

import torch
from .metrics import spectral_beta, multifractal_spectrum, graph_fractal_dim, functional_connectivity_latents

def l_spec_1overf(
    latents: torch.Tensor, fs: float, band: Tuple[float,float], beta_target: float, weight: float
) -> torch.Tensor:
    beta = spectral_beta(latents, fs, band)  # (B, C) or (...)
    return weight * torch.mean((beta - beta_target)**2)

def l_multifractal(
    latents: torch.Tensor, q_vals: torch.Tensor, scales: torch.Tensor, weight: float, tau_target: Optional[torch.Tensor]=None
) -> torch.Tensor:
    mf = multifractal_spectrum(latents, q_vals, scales)
    # Encourage smooth τ(q) curvature & optionally match target τ(q)
    tau = mf['tau']
    smooth = torch.mean((tau[...,2:] - 2*tau[...,1:-1] + tau[...,:-2])**2)
    match = 0.0 if tau_target is None else torch.mean((tau - tau_target)**2)
    return weight * (smooth + 0.1*match)

def l_graph_fd(
    latents: torch.Tensor, eps_values: torch.Tensor, thresh: float, D_target: Optional[float], weight: float
) -> torch.Tensor:
    adj = functional_connectivity_latents(latents, "pearson", thresh=thresh)
    D = graph_fractal_dim(adj, eps_values)  # (B,)
    if D_target is None:
        return weight * torch.var(D)        # stabilize across batches
    return weight * torch.mean((D - D_target)**2)

probes.py 

# probes.py

def latent_fd_probe(latents: torch.Tensor, cfg) -> Dict[str, torch.Tensor]:
    """Return dict of FD/β/H summary stats for logging."""
    ...

def attention_fd_coupling(attn_weights: torch.Tensor, series_fd: torch.Tensor) -> torch.Tensor:
    """
    Correlate attention maps with local FD changes across time/heads.
    Returns coupling coefficients per head/layer.
    """

def causal_scale_ablation(
    x: torch.Tensor, mode: str = "high_beta_only", ratio: float = 0.15, fs: float = 1000., band=(2,80)
) -> torch.Tensor:
    """
    FFT-domain mask: zero top-ratio 1/f-contributing bins ("high_beta") or low-frequency bins.
    Returns ablated x (same shape).
    """

Lightning callbacks in trainers/callbacks_fractals.py:

On train/val step end: compute/log_dict FD/β/H means/variances.

Every tb_hist_every_n: histogram of β/H across channels, scatter of attention-FD coupling.

Optional: run ablation on a validation mini-batch and log Δmetrics & Δloss.

stimuli.py
Fractal generators (CPU okay; cache to tensors).
def fbm_series(n_samples: int, beta: float, fs: float) -> torch.Tensor:
    """Fractal Brownian motion via spectral synthesis with 1/f^β."""

def one_over_f_noise(n_samples: int, beta: float, fs: float) -> torch.Tensor:
    """Colored noise generator (β=0 white, 1 pink, 2 brown)."""

def cascade_multifractal(n: int, depth: int = 8, p: float = 0.6) -> torch.Tensor:
    """Multiplicative cascade; returns multifractal measure/time-series."""

simulators.py
def fractional_ou(n_samples: int, H: float, theta: float, sigma: float, fs: float) -> torch.Tensor:
    """
    Fractional OU via frequency-domain shaping; differentiable w.r.t H, theta, sigma.
    """
def dendrite_growth(target_D: float, steps: int = 200, seed: int = 0) -> Dict[str, torch.Tensor]:
    """
    Toy arbor growth that adjusts branching to match target fractal dimension.
    Returns coordinates/edges; offline use for analysis/visuals; later: couple to token features.
    """

7) Backbone hooks

A) Fractional kernels in Mamba/SSM

Add optional power-law impulse responses to SSM/state-mix convs:

Generate taps h[t] ∝ (t+1)^(-α), convolve via 1D depthwise conv (fast via FFT for long taps).

α sampled per layer within [min_alpha, max_alpha] or learned via softplus param (bounded).

Gate between standard kernel and fractional kernel (learnable λ).

Where: Extend your SSM/mixer module, e.g. models/backbones/ssm_mamba.py:
class FractionalKernel(nn.Module):
    def __init__(self, n_taps, min_alpha, max_alpha, channels):
        ...
    def forward(self, x):  # x: (B,C,T)
        # build h (B,C,Taps) from alphas; depthwise conv x * h
        return y

Wire with config: fractals.backbone.fractional_kernels.enable.

B) Scale encodings

Prepend wavelet scattering or simpler log-spaced filterbank energies as extra features per token.

Implement minimalistic: multi-rate average pooling + band-energy (no heavy deps), or optional kymatio if available.

Where: tokenization layer (data/tokenizers/*) and PopT/Perceiver input packing.

8) Trainer integration

In your LightningModule (or equivalent), add:

# forward() stays unchanged

def training_step(...):
    out = self.model(batch)
    loss = out["loss"]

    if cfg.fractals.enable and global_step > cfg.fractals.priors.warmup_steps:
        latents = out["latents"]            # ensure model returns these
        fs = cfg.fractals.metrics.sample_rate_hz
        band = tuple(cfg.fractals.metrics.psd_freq_band)

        loss += l_spec_1overf(latents, fs, band,
                              cfg.fractals.priors.spectral_beta_target,
                              cfg.fractals.priors.lambda_spec)

        q = torch.tensor(cfg.fractals.multifractal.q_vals, device=latents.device, dtype=latents.dtype)
        scales = torch.tensor(cfg.fractals.multifractal.scales, device=latents.device)

        loss += l_multifractal(latents, q, scales, cfg.fractals.priors.lambda_mf)

        eps = torch.tensor(cfg.fractals.graph.box_eps, device=latents.device)
        loss += l_graph_fd(latents, eps, cfg.fractals.graph.corr_threshold,
                           D_target=None, weight=cfg.fractals.priors.lambda_graphfd)

    self.log("train/loss", loss)
    return loss

Also register callbacks from callbacks_fractals.py in your trainer construction.

9) Logging & plots: mechanistic/fractals/plotting.py

plot_fd_hist(fd_vals, title)

plot_beta_time(beta_seq)

plot_attention_fd_coupling(couplings_by_head)

Prefer matplotlib; export to TensorBoard/W&B images.

10) Tests (high-value)

test_fractals_metrics.py

HF/DFA/β on synthetic signals with known exponents (white/pink/brown noise).

Multifractal spectrum shape sanity.

Graph FD monotonicity across eps.

test_fractals_regularizers.py

Loss decreases toward target on toy latent tensors.

test_fractals_probes.py

Scale ablation reduces high-freq energy.

test_fractals_stimuli.py

Generated series achieve target β within tolerance.

test_fractals_simulators.py

fOU H estimate ~ requested H.

Keep CPU runtime per test < 2s.

11) Example notebook / script

Add examples/fractals_quickstart.ipynb (or .py) that:

Generates fBm signals with β∈{0.5,1.0,1.5}.

Computes HF/DFA/β; plots distributions.

Runs a tiny model (Perceiver stub) on synthetic data with and without 1/f prior; compares β drift and validation loss.

Demonstrates attention–FD coupling on dummy attention maps.

12) Docs: docs/methods_fractals.md

Explain succinctly:

What each metric estimates (HF, DFA/α, H, β, τ(q), graph FD).

Why these matter for neural data (scale-free structure; state/development links).

How priors are applied; safe defaults; how to disable.

How to interpret logs/plots and run ablations.
13) PR plan / milestones

PR-1: Scaffolding & metrics

Add directory + metrics implementations + tests + docs stub.

PR-2: Regularizers & trainer hooks

Implement priors; integrate config/flags; add callbacks; tests.

PR-3: Stimuli & simulators

Generators + fOU + dendrite growth toy; examples; tests.

PR-4: Backbone fractional & scale encodings

Fractional kernels in SSM; scale features for tokens; benchmarks.

PR-5: Dashboards & examples

Plotting helpers; quickstart; W&B panels.

Ship each PR with green tests and a short CHANGELOG entry.

14) Benchmarks (acceptance)

Speed: Metrics pass on 32×(T=8192,C=256) batch < 300ms on A100.

Numerics: On synthetic signals, β estimate within ±0.1 across bands; DFA α within ±0.05.

Training: With lambda_spec=0.05, latent β converges within [0.8,1.2] without hurting task loss (>99% baseline).

Ablation: “high_beta_only” ablation causes >X% drop on tasks known to rely on fine temporal features (define X per dataset).

Stability: Graph FD variance reduced vs. baseline across seeds (if l_graphfd active).

16) Integration checklist (Claude, follow in order)

Create directories & files exactly as above.

Implement metric functions (temporal → multifractal → graph).

Add regularizers and unit tests for all metrics/regularizers.

Wire config loading and default neurofm_fractals.yaml.

Insert Lightning callback into the trainer creation path.

Modify model forward to return latents (Perceiver latents + PopT tokens + SSM states, or at least one stable latent representation).

Implement fractional kernels and scale encodings (guarded by config).

Add stimuli/simulators + example script.

Add plotting helpers + W&B panels (hist β/H, coupling heatmaps).

Write docs/methods_fractals.md.

Run tests & a 5-minute smoke train with priors enabled.