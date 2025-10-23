# prepare_full_dataset.py (Simplified)

import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from speed_train3070 import load_allen_dataset
# Re-use the Config class from scaleup_benchmark.py (or a similar one)
# Re-use the load_allen_dataset and _process_session functions from the original script

# --- NEW DATA PREPARATION SCRIPT ---

class DataPrepConfig:
    data_dir = Path("./data/allen_neuropixels")
    processed_data_dir = data_dir / "processed_sequences_full"
    allen_cache_dir = data_dir / "cache"

    sequence_length = 100
    bin_size_ms = 10.0
    bin_size_sec = 10.0 / 1000.0
    max_units = 384 # Use the true max unit size for storage
    train_split = 0.8
def _process_session(self, session):
        """Process session into training sequences."""
        units = session.units
        spike_times = session.spike_times

        # Use the aggressively reduced max_units
        if len(units) > self.max_units:
            unit_ids = units.index[:self.max_units].tolist()
        else:
            unit_ids = units.index.tolist()

        n_units = len(unit_ids)

        max_time = 0
        for unit_id in unit_ids:
            if unit_id in spike_times:
                times = spike_times[unit_id]
                if len(times) > 0:
                    max_time = max(max_time, times.max())

        if max_time == 0:
            return []

        n_bins = int(max_time / self.bin_size_sec)
        time_bins = np.linspace(0, max_time, n_bins + 1)

        binned_spikes = np.zeros((n_bins, n_units), dtype=np.float32)

        for i, unit_id in enumerate(unit_ids):
            if unit_id in spike_times:
                times = spike_times[unit_id]
                if len(times) > 0:
                    bin_indices = np.digitize(times, time_bins) - 1
                    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
                    for bin_idx in bin_indices:
                        binned_spikes[bin_idx, i] += 1

        sequences = []
        stride = self.sequence_length // 2

        for start_idx in range(0, len(binned_spikes) - self.sequence_length, stride):
            end_idx = start_idx + self.sequence_length
            seq_spikes = binned_spikes[start_idx:end_idx]

            if seq_spikes.sum() < 10:
                continue

            sequences.append({
                'spikes': seq_spikes,
                'n_units': n_units,
            })

        return sequences

def process_and_save_all_sessions():
    config = DataPrepConfig()
    config.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading Allen dataset cache...")
    # Assume load_allen_dataset and EcephysProjectCache are defined/imported
    cache, all_session_ids = load_allen_dataset(config) 

    print(f"Starting to process and save {len(all_session_ids)} sessions...")
    
    all_sequences = []

    for session_id in tqdm(all_session_ids, desc="Processing sessions"):
        try:
            session = cache.get_session_data(session_id)
            print(f"Processing session {session_id} with {len(session.units)} units.")
            
            sequences = _process_session(config, session)
            
            # Save sequences for this session to a file
            session_path = config.processed_data_dir / f"session_{session_id}.npz"
            
            # Structure sequences for saving (list of dicts to a single array)
            spikes_list = [seq['spikes'] for seq in sequences]
            if spikes_list:
                # Pad to max_units (384) before saving
                max_u = max(s.shape[1] for s in spikes_list)
                padded_spikes = []
                for s in spikes_list:
                     pad_width = config.max_units - s.shape[1]
                     padded = np.pad(s, ((0, 0), (0, pad_width)), 'constant')
                     padded_spikes.append(padded)

                np.savez_compressed(
                    session_path, 
                    spikes=np.array(padded_spikes), 
                    n_sequences=len(spikes_list)
                )
                print(f"   -> Saved {len(spikes_list)} sequences to {session_path.name}")
            
            all_sequences.extend(spikes_list)

        except Exception as e:
            print(f"Error processing session {session_id}: {e}")
            continue

    print(f"\n✅ Data Preparation Complete. Total sequences generated: {len(all_sequences)}")
    print(f"Data stored in: {config.processed_data_dir}")

if __name__ == '__main__':
    process_and_save_all_sessions()