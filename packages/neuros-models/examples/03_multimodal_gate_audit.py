"""Learn multimodal gates, then distinguish gate values from causal dependence."""

import numpy as np

from neuros.models import AttentionFusionModel

rng = np.random.default_rng(7)
eeg = rng.normal(size=(96, 6)).astype("float32")
emg = rng.normal(size=(96, 4)).astype("float32")
imu = rng.normal(size=(96, 3)).astype("float32")
y = (2.0 * eeg[:, 0] + 0.2 * emg[:, 0] > 0).astype(int)
X = np.concatenate([eeg, emg, imu], axis=1)

model = AttentionFusionModel(
    [6, 4, 3],
    n_classes=2,
    fusion_dim=16,
    n_epochs=5,
    batch_size=24,
    random_state=7,
)
model.train(X, y)
weights = model.get_attention_weights(X[:8])
print("mean learned gate weights [EEG, EMG, IMU]:", weights.mean(axis=0))

baseline = model.predict_proba(X[:16])[:, 1]
X_without_eeg = X[:16].copy()
X_without_eeg[:, :6] = 0.0
counterfactual = model.predict_proba(X_without_eeg)[:, 1]
print("mean probability change after EEG suppression:", float(np.abs(baseline - counterfactual).mean()))
print("The suppression effect is causal for this intervention; the raw gate weight alone is not.")
