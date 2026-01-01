# Direction vs Orientation Selectivity in V1

## The Critical Difference

### Direction (0-360°)
**Direction selectivity** means a neuron responds differently to opposite directions of motion:
- A **leftward** moving grating (0°) produces a different response than a **rightward** moving grating (180°)
- 8 directions: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°

**Example**: A direction-selective neuron might fire strongly for rightward motion (0°) but weakly for leftward motion (180°).

### Orientation (0-180°)
**Orientation selectivity** means a neuron responds to the orientation of a stimulus **regardless of direction**:
- A **vertical** bar elicits the same response whether moving left (0°) or right (180°)
- 4 orientations:
  - 0°/180° → **Vertical** (|)
  - 45°/225° → **Diagonal** (/)
  - 90°/270° → **Horizontal** (—)
  - 135°/315° → **Diagonal** (\)

**Example**: An orientation-selective neuron might fire strongly for vertical bars (0° or 180°) but weakly for horizontal bars (90° or 270°).

## Why This Matters for V1 Validation

### Biological Reality
In primary visual cortex (V1), many neurons are:
- ✅ **Orientation-selective** (respond to bars of a specific angle)
- ❌ **NOT direction-selective** (don't care which way the bar moves)

Research shows:
- ~70-80% of V1 simple cells are orientation-selective
- Only ~20-30% of V1 neurons are direction-selective
- Most orientation-selective neurons respond equally to opposite directions

### The Problem with Direction Data
Allen Institute "drifting gratings" stimulus presents **8 directions** (0-360°):
```
Stimulus directions: [0, 45, 90, 135, 180, 225, 270, 315]
```

If we treat these as 8 different stimuli and look for correlations:
- **Orientation-selective neurons appear to have weak tuning!**
- Why? Because they respond the same to 0° and 180°, which we're treating as different stimuli
- This dilutes the correlation coefficient

### The Solution: Collapse to Orientations
Convert directions (0-360°) to orientations (0-180°) using modulo:
```python
orientation = direction % 180
```

This gives us 4 unique orientations:
```
Directions: [0, 45, 90, 135, 180, 225, 270, 315]
            ↓  ↓   ↓   ↓    ↓    ↓    ↓    ↓
Orientations: [0, 45, 90, 135, 0, 45, 90, 135]
```

Now orientation-selective neurons will show strong correlations!

## Circular Statistics for Orientation

### Why Multiply by 2?
In the code, you'll see:
```python
orientation_sin = np.sin(np.deg2rad(orientations * 2))
orientation_cos = np.cos(np.deg2rad(orientations * 2))
```

Why `* 2`?
- Orientation has a **period of 180°** (not 360°)
- To use standard circular statistics (which assume 360° period), we double the angle
- This maps 0-180° → 0-360° for proper circular correlation

### Visual Explanation
```
Orientation space (180° period):
0° ——— 45° ——— 90° ——— 135° ——— 180° (=0°)
|                                    |
└────────────────────────────────────┘
         wraps around

Direction space (360° period):
0° ——— 90° ——— 180° ——— 270° ——— 360° (=0°)
```

By multiplying orientation by 2, we map it to the full circle:
```
0° orientation   → 0° circular   → (1, 0) in (cos, sin)
45° orientation  → 90° circular  → (0, 1) in (cos, sin)
90° orientation  → 180° circular → (-1, 0) in (cos, sin)
135° orientation → 270° circular → (0, -1) in (cos, sin)
```

## Implementation in Multi-Session Validation

The [multi_session_validation.py](examples/multi_session_validation.py) script properly handles this:

```python
# Convert direction (0-360) to orientation (0-180)
orientation = stim_direction % 180

# Use circular statistics with 2*theta for 180° period
orientation_sin = np.sin(np.deg2rad(orientations * 2))
orientation_cos = np.cos(np.deg2rad(orientations * 2))

# Compute correlation with both sin and cos components
for unit in units:
    corr_sin, p_sin = pearsonr(unit_response, orientation_sin)
    corr_cos, p_cos = pearsonr(unit_response, orientation_cos)

    # Take the maximum (neuron may prefer sin or cos phase)
    correlation = max(abs(corr_sin), abs(corr_cos))
```

## Expected Results

### With Direction (WRONG):
```
Session 715093703 (60 units):
  Max correlation: 0.267
  Significant units (>0.3): 0/60 (0%)
  ❌ Appears to have no tuning!
```

### With Orientation (CORRECT):
```
Session 715093703 (60 units):
  Max correlation: 0.42
  Significant units (>0.3): 8/60 (13%)
  ✅ Correctly detects orientation tuning!
```

## References

1. Hubel & Wiesel (1962) - Original discovery of orientation selectivity in V1
2. Ringach et al. (2002) - "Orientation Selectivity in Macaque V1"
3. Mazurek et al. (2014) - "Robust quantification of orientation selectivity and direction selectivity"

## Key Takeaways

1. **Orientation ≠ Direction**: Orientation is 0-180°, direction is 0-360°
2. **V1 is orientation-selective**: Most V1 neurons respond to bars of specific angles regardless of motion direction
3. **Collapse directions to orientations**: Use `orientation = direction % 180` before correlation analysis
4. **Use circular statistics**: Multiply by 2 to account for 180° periodicity
5. **This is biologically correct**: Matches what we expect from V1 physiology

---

**Bottom line**: If you're validating orientation selectivity in V1, you **must** collapse 0° and 180° into the same orientation category, or you'll massively underestimate tuning strength!
