# Data Acquisition Scripts for NeuroFMx

This directory contains scripts to download and preprocess multimodal neural data for training NeuroFMx.

## Available Scripts

### 1. IBL Dataset (`download_ibl.py`)
Downloads spike and behavioral data from International Brain Laboratory.

```bash
python download_ibl.py --n_sessions 30 --output_dir ./data/ibl/processed
```

**Requirements:** `pip install ONE-api ibllib`

### 2. Allen 2-Photon Calcium Imaging (`download_allen_2p.py`)
Downloads calcium imaging data from Allen Brain Observatory.

```bash
python download_allen_2p.py --n_experiments 15 --output_dir ./data/allen_2p/processed
```

**Requirements:** `pip install allensdk`

### 3. Human EEG (`download_eeg.py`)
Downloads EEG motor imagery data from PhysioNet.

```bash
python download_eeg.py --n_subjects 20 --output_dir ./data/eeg/processed
```

**Requirements:** `pip install mne`

### 4. fMRI (`download_fmri.py`)
Processes fMRI data with ROI parcellation.

```bash
python download_fmri.py --output_dir ./data/fmri/processed --n_rois 400
```

**Requirements:** `pip install nilearn nibabel`

## Data Format

All scripts output `.npz` files with the following structure:

```python
{
    '<modality>': np.ndarray,  # Main neural data (T, N) or (T, C)
    'behavior': np.ndarray,     # Behavioral variables (T, B)
    'metadata': dict            # Session/trial info
}
```

## Installation

Install all dependencies:

```bash
pip install ONE-api allensdk mne nilearn nibabel pynwb
```

## Output Directory Structure

```
data/
├── ibl/
│   └── processed/
│       ├── train/
│       ├── val/
│       └── test/
├── allen_2p/
│   └── processed/
│       ├── train/
│       ├── val/
│       └── test/
...
```

## Next Steps

After downloading data:
1. Verify data integrity
2. Train multimodal tokenizers
3. Begin model training
