# Mechanistic Interpretability Hooks - Quick Reference

## 1-Minute Setup

```python
# Install
pip install pytorch-lightning

# Import
from neuros_neurofm.interpretability import MechIntCallback, MechIntConfig

# Configure
config = MechIntConfig(
    sample_layers=['mamba_backbone.blocks.3', 'popt'],
    save_hidden_every_n_steps=200
)

# Train
trainer = pl.Trainer(callbacks=[MechIntCallback(config)])
trainer.fit(model, dataloader)

# Activations saved to: ./mechint_cache/
```

## Quick Reference Table

| Task | Code |
|------|------|
| **Basic Setup** | `config = MechIntConfig(sample_layers=['layer1'])` |
| **Lightning Integration** | `trainer = pl.Trainer(callbacks=[MechIntCallback(config)])` |
| **Manual Hooks** | `hooks = MechIntHooks(config); hooks.register_hooks(model)` |
| **Save Activations** | `sampler.save_activations(global_step=100)` |
| **Run Analysis** | `runner = EvalMechIntRunner(model, config); runner.run_mechint_eval(...)` |
| **Export Results** | `runner.export_results('./results')` |
| **Local Storage** | `MechIntConfig(storage_backend='local')` |
| **S3 Storage** | `MechIntConfig(storage_backend='s3', s3_bucket='my-bucket')` |
| **Add API Endpoints** | `mixin = FastAPIIntegrationMixin(model, config); mixin.add_routes(app)` |

## Configuration Cheat Sheet

```python
MechIntConfig(
    # Required
    sample_layers=['layer1', 'layer2'],

    # Common
    save_hidden_every_n_steps=200,        # How often to save
    storage_backend='local',               # 'local', 's3', 'both'
    storage_path='./mechint_cache',        # Where to save

    # Analysis
    analyses_to_run=['sae', 'neuron'],     # Which analyses

    # S3 (if using)
    s3_bucket='my-bucket',
    s3_prefix='exp001/activations',

    # Performance
    max_activations_per_shard=10000,       # Samples per file
    verbose=True                           # Logging
)
```

## Usage Patterns

### Pattern 1: Training with Lightning (Easiest)

```python
trainer = pl.Trainer(
    callbacks=[MechIntCallback(config)],
    max_epochs=100
)
trainer.fit(model, dataloader)
```

### Pattern 2: Manual Training Loop

```python
hooks = MechIntHooks(config)
hooks.register_hooks(model)

for step, batch in enumerate(dataloader):
    outputs = model(batch)
    # ... training code ...

    if step % config.save_hidden_every_n_steps == 0:
        hooks.sampler.save_activations(step)
        hooks.sampler.clear_cache()
```

### Pattern 3: Evaluation Only

```python
runner = EvalMechIntRunner(model, config)
results = runner.run_mechint_eval(
    checkpoint_path='./model.pt',
    hidden_shards_path='./mechint_cache'
)
runner.export_results('./results')
```

### Pattern 4: API Server

```python
from fastapi import FastAPI

app = FastAPI()
mixin = FastAPIIntegrationMixin(model, config)
mixin.add_routes(app)

# Endpoints: /interpret, /interpret/upload, /interpret/layers
```

## Common Commands

```bash
# List captured layers
ls mechint_cache/

# Check manifest
cat mechint_cache/manifest.json

# Load a shard (Python)
data = torch.load('mechint_cache/activations_shard_000000_step_0.pt')
print(data['activations'].keys())

# Run tests
pytest tests/test_mechint_hooks.py -v

# Start API server with mech-int
uvicorn app:app --reload
```

## API Endpoints

```bash
# List available layers
curl http://localhost:8000/interpret/layers

# Run neuron analysis
curl -X POST http://localhost:8000/interpret \
  -H "Content-Type: application/json" \
  -d '{"analysis_type": "neuron", "layer_name": "layer1"}'

# Upload and analyze
curl -X POST http://localhost:8000/interpret/upload \
  -F "file=@activations.pt" \
  -F "analysis_type=feature"
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No activations captured | Check layer names with `for name, _ in model.named_modules(): print(name)` |
| Out of memory | Reduce `max_activations_per_shard` or `save_hidden_every_n_steps` |
| S3 upload fails | Check AWS credentials: `aws s3 ls s3://my-bucket/` |
| Analysis fails | Verify shards exist: `ls mechint_cache/activations_*.pt` |
| Training slow | Increase `save_hidden_every_n_steps` or reduce tracked layers |

## File Structure

```
mechint_cache/
├── activations_shard_000000_step_0.pt      # Activations from step 0
├── activations_shard_000001_step_200.pt    # Activations from step 200
├── activations_shard_000002_step_400.pt    # Activations from step 400
└── manifest.json                            # Index of all shards

results/
├── mechint_results.json                     # Full analysis results
└── mechint_report.md                        # Human-readable report
```

## Analysis Types

| Type | What it Does | Output |
|------|-------------|--------|
| `sae` | Trains sparse autoencoders | Feature dictionaries, sparsity stats |
| `neuron` | Analyzes neuron activations | Mean, max, sparsity per neuron |
| `feature` | PCA and clustering | Explained variance, clusters |
| `circuit` | Circuit discovery | Intervention effects |
| `causal` | Causal importance | Ablation results |

## Best Practices

1. ✅ Start with `save_hidden_every_n_steps=500` for development
2. ✅ Use `storage_backend='s3'` for long training runs
3. ✅ Track 2-3 layers max during initial experiments
4. ✅ Run analysis offline, not during training
5. ✅ Keep manifests for reproducibility
6. ✅ Monitor disk space regularly
7. ✅ Use descriptive `s3_prefix` for experiment tracking

## Quick Examples

### Example 1: Minimal Setup

```python
from neuros_neurofm.interpretability import MechIntCallback

trainer = pl.Trainer(
    callbacks=[MechIntCallback(config={'sample_layers': ['layer1']})]
)
```

### Example 2: Full Pipeline

```python
# 1. Train with hooks
config = MechIntConfig(sample_layers=['layer1', 'layer2'])
trainer = pl.Trainer(callbacks=[MechIntCallback(config)])
trainer.fit(model, dataloader)

# 2. Analyze
runner = EvalMechIntRunner(model, config)
results = runner.run_mechint_eval(hidden_shards_path='./mechint_cache')

# 3. Export
runner.export_results('./results')
```

### Example 3: Custom Sampler

```python
sampler = ActivationSampler(
    layers=['layer1'],
    save_dir='./custom_cache'
)
sampler.register_hooks(model)

# Training loop
for batch in dataloader:
    output = model(batch)

# Save
sampler.save_activations(global_step=100)
```

## Layer Name Examples

Common NeuroFMX layer names to track:

```python
sample_layers=[
    'mamba_backbone.blocks.0',        # First Mamba block
    'mamba_backbone.blocks.3',        # Middle Mamba block
    'mamba_backbone.blocks.7',        # Last Mamba block
    'perceiver',                       # Perceiver fusion layer
    'popt',                            # PopT aggregator
    'heads.decoder',                   # Decoder head
    'heads.encoder'                    # Encoder head
]
```

## Resources

- **Full Guide:** `docs/mechint_hooks_guide.md`
- **Examples:** `examples/mechint_hooks_example.py`
- **Tests:** `tests/test_mechint_hooks.py`
- **Source:** `src/neuros_neurofm/interpretability/hooks.py`

---

**Get Started:** `from neuros_neurofm.interpretability import MechIntCallback`
