"""Attach SourceWeigher reliability to a native neurOS fusion node."""
from __future__ import annotations

from neuros.runtime import NodeKind, RuntimeNode
from neuros_sourceweigher import ReliabilityWeightedFusion

# In a real RuntimeGraph these keys are the upstream transform node IDs.
fusion = ReliabilityWeightedFusion(
    {
        "transform:eeg": 0.65,
        "transform:emg": 0.25,
        "transform:imu": 0.10,
    },
    mode="scale_concat",
)

fusion_node = RuntimeNode(
    "fusion:reliability",
    NodeKind.FUSION,
    fusion,
)

print(fusion_node)
print("current reliability:", fusion.weights)

# A quality/drift controller can update the operator without rebuilding its
# estimator. RuntimeGraph wiring remains owned by neuros-core.
fusion.set_weights(
    {
        "transform:eeg": 0.25,
        "transform:emg": 0.55,
        "transform:imu": 0.20,
    }
)
print("after EEG degradation:", fusion.weights)
