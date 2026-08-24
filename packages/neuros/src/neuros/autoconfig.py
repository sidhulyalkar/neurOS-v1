"""Compatibility auto-configuration for the composed neurOS SDK.

This is intentionally a user-facing composition helper, not a kernel feature:
it selects concrete drivers and models and therefore belongs in the ``neuros``
meta-distribution, which depends on those packages. New applications should
prefer versioned configuration plus plugin discovery.
"""

from __future__ import annotations

from typing import Optional

from neuros.pipeline import Pipeline


def generate_pipeline_for_task(
    task_description: str,
    *,
    use_brainflow: bool = False,
    model_name: Optional[str] = None,
    fs: float = 250.0,
    channels: int = 8,
) -> Pipeline:
    try:
        from neuros.agents import BlinkAgent, CalciumAgent, FacialAgent, MotionAgent, PoseAgent, VideoAgent
        from neuros.drivers.brainflow_driver import BrainFlowDriver
        from neuros.drivers.calcium_imaging_driver import CalciumImagingDriver
        from neuros.drivers.dataset_driver import DatasetDriver
        from neuros.drivers.ecog_driver import ECoGDriver
        from neuros.drivers.emg_driver import EMGDriver
        from neuros.drivers.eog_driver import EOGDriver
        from neuros.drivers.mock_driver import MockDriver
        from neuros.drivers.motion_sensor_driver import MotionSensorDriver
        from neuros.drivers.video_driver import VideoDriver
        from neuros.models import (
            CNNModel,
            DinoV3Model,
            EEGNetModel,
            GBDTModel,
            KNNModel,
            RandomForestModel,
            SVMModel,
            SimpleClassifier,
            TransformerModel,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Auto-configuration requires the neurOS driver and model packages. "
            "Install the BCI profile before calling generate_pipeline_for_task()."
        ) from exc

    desc = task_description.lower()
    name = model_name.lower() if model_name else ""
    if not name:
        if "transformer" in desc or "sequence" in desc or "transform" in desc:
            name = "transformer"
        elif "ssvep" in desc or "steady state" in desc:
            name = "eegnet"
        elif "cnn" in desc or "convolution" in desc:
            name = "cnn"
        elif "motor" in desc and "imagery" in desc:
            name = "svm"
        elif "regression" in desc:
            name = "gbdt"
        else:
            name = "random_forest"

    factories = {
        "eegnet": EEGNetModel,
        "cnn": CNNModel,
        "transformer": TransformerModel,
        "dino": DinoV3Model,
        "dino_v3": DinoV3Model,
        "random_forest": RandomForestModel,
        "svm": SVMModel,
        "knn": KNNModel,
        "gbdt": GBDTModel,
        "simple": SimpleClassifier,
    }
    if name not in factories:
        raise ValueError(f"Unknown model name: {name}")
    model = factories[name]()

    bands = None
    if "ssvep" in desc:
        bands = {"alpha_beta": (8.0, 20.0)}
    elif "motor" in desc:
        bands = {"mu_beta": (8.0, 30.0)}

    processing_agent_class = None
    processing_kwargs = {}
    dataset_names = ["iris", "digits", "wine", "breast cancer", "breast_cancer", "cancer"]

    if any(key in desc for key in ["dataset", "data set", "reprocess", "reanalysis"]):
        selected = next((candidate for candidate in dataset_names if candidate in desc), "iris")
        selected_name = "breast_cancer" if selected in ("breast cancer", "cancer") else selected
        driver = DatasetDriver(dataset_name=selected_name, sampling_rate=fs)
        processing_agent_class = MotionAgent
    elif any(key in desc for key in ["calcium", "imaging", "optical", "fluorescence", "two-photon"]):
        driver = CalciumImagingDriver(frame_rate=fs)
        processing_agent_class = CalciumAgent
        if not model_name and name in ["random_forest", "knn", "svm"]:
            model = CNNModel()
    elif any(key in desc for key in ["video", "pose", "facial", "blink", "face"]):
        driver = VideoDriver(frame_rate=fs, resolution=(64, 64), channels=3)
        if "pose" in desc:
            processing_agent_class = PoseAgent
        elif any(key in desc for key in ["facial", "face"]):
            processing_agent_class = FacialAgent
        elif "blink" in desc:
            processing_agent_class = BlinkAgent
        else:
            processing_agent_class = VideoAgent
    elif any(key in desc for key in ["motion", "imu", "movement", "acceleration"]):
        driver = MotionSensorDriver(sampling_rate=fs)
        processing_agent_class = MotionAgent
    elif any(key in desc for key in ["ecog", "corticography", "cortical", "intracranial"]):
        driver = ECoGDriver(sampling_rate=fs, channels=channels)
        if not model_name and isinstance(model, RandomForestModel):
            model = CNNModel()
    elif any(key in desc for key in ["emg", "muscle", "myography"]):
        driver = EMGDriver(sampling_rate=fs, channels=channels)
    elif any(key in desc for key in ["eog", "ocular", "eye"]):
        driver = EOGDriver(sampling_rate=fs, channels=channels)
        if not model_name and isinstance(model, RandomForestModel):
            model = KNNModel()
    else:
        driver = (
            BrainFlowDriver(board_id=0, sampling_rate=fs, channels=channels)
            if use_brainflow
            else MockDriver(sampling_rate=fs, channels=channels)
        )

    return Pipeline(
        driver=driver,
        model=model,
        fs=fs,
        bands=bands,
        adaptation=True,
        processing_agent_class=processing_agent_class,
        processing_kwargs=processing_kwargs,
    )
