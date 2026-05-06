# Astro Integration Demo Fixes

## Summary

Fixed multiple issues in the astrocyte integration demo to make it compatible with the neuros-astro token format.

## Issues Fixed

### 1. Token File Key Mismatch
**Problem:** Demo expected `event_tokens`, `timestamps`, `metadata` but actual file has:
- `tokens` (not `event_tokens`)
- `timestamps_s` (not `timestamps`)
- `metadata_json` (not `metadata`)

**Fix:** Updated [astro_integration_demo.py](examples/astro_integration_demo.py#L51-L67) to use correct keys.

### 2. PerceiverIO Parameter Mismatch
**Problem:** `TypeError: PerceiverIO.__init__() got an unexpected keyword argument 'num_latents'`

**Fix:** Updated [multimodal_neurofmx.py](src/neuros_neurofm/models/multimodal_neurofmx.py#L141-L148) to use:
- `n_latents` (not `num_latents`)
- `n_layers` (not `num_cross_attention_layers`)
- `n_heads` (not separate widening factors)

### 3. MambaBackbone Parameter Mismatch
**Problem:** `TypeError: MambaBackbone.__init__() got an unexpected keyword argument 'n_layers'`

**Fix:** Updated [multimodal_neurofmx.py](src/neuros_neurofm/models/multimodal_neurofmx.py#L151-L157) to use:
- `n_blocks` (not `n_layers`)

### 4. Region IDs are Strings
**Problem:** `region_ids` contains strings like 'roi_002', 'roi_007' - can't convert directly to tensors

**Fix:** Added string-to-integer mapping in both demos:
```python
if region_ids_raw.dtype == object:
    unique_regions = np.unique(region_ids_raw)
    region_to_idx = {region: idx for idx, region in enumerate(unique_regions)}
    region_ids = np.array([region_to_idx[r] for r in region_ids_raw], dtype=np.int64)
```

### 5. Tokenizer Returns Tuple
**Problem:** `from_neuros_astro_tokens()` and `forward()` both return tuples, not single tensors

**Fix:** Updated both demos to properly unpack:
```python
# from_neuros_astro_tokens returns (event_tensor, timestamp_tensor)
event_tensor, timestamp_tensor = tokenizer.from_neuros_astro_tokens(...)

# forward with return_sequence=False returns (tokens, mask)
astro_input, mask = tokenizer(event_tensor, return_sequence=False)
```

## New Files Created

### [astro_integration_demo_simple.py](examples/astro_integration_demo_simple.py)
A simplified version that doesn't require the full MultiModalNeuroFMX model or mamba-ssm package. Perfect for testing the tokenization pipeline.

## Running the Demos

### Simple Demo (No Dependencies Required)
```bash
cd packages/neuros-neurofm
python examples/astro_integration_demo_simple.py \
    --astro-tokens ../neuros-astro/allen_processed/2p_session_545446482/astro_tokens.npz \
    --max-events 50
```

**Output:**
- Loads astro tokens from neuros-astro
- Creates AstroTokenizer
- Converts events to model-ready tensors
- Shows token statistics

### Full Demo (Requires mamba-ssm)
```bash
# Install dependency first
pip install mamba-ssm

# Run full demo
cd packages/neuros-neurofm
python examples/astro_integration_demo.py \
    --astro-tokens ../neuros-astro/allen_processed/2p_session_545446482/astro_tokens.npz \
    --max-events 50
```

**Output:**
- Everything from simple demo
- Creates full MultiModalNeuroFMX model
- Runs forward pass through model
- Extracts latent representations

## Key Learnings

### neuros-astro Token Format
```
astro_tokens.npz contains:
- tokens: (n_events, n_features) - event feature array
- timestamps_s: (n_events,) - event times in seconds
- region_ids: (n_events,) - string identifiers like 'roi_002'
- metadata_json: JSON string with session info
- feature_names: array of feature names
- session_id: session identifier
```

### Feature List (10 features)
1. onset_time
2. duration_s
3. peak_dff
4. area_px
5. centroid_y
6. centroid_x
7. propagation_speed
8. direction_sin
9. direction_cos
10. confidence

## Next Steps

1. Train MultiModalNeuroFMX with both astro and neural modalities
2. Run ablation studies: with vs without astro
3. Analyze astro contribution to model performance
4. Extend to multi-session datasets
