"""
Automatic pipeline configuration for neurOS.

This module defines simple heuristics to assemble a neurOS pipeline based on
a free‑form task description and optional user constraints.  It acts as
an early demonstration of how intelligent agents can remove manual setup
steps from brain–computer interface development.

The core function :func:`generate_pipeline_for_task` inspects keywords in the
task description to choose appropriate frequency bands and models.  In the
future, this logic could be replaced with more sophisticated natural
language processing and AutoML techniques.
"""

from __future__ import annotations

import re
from typing import Optional

from .pipeline import Pipeline
from .drivers.mock_driver import MockDriver
from .drivers.brainflow_driver import BrainFlowDriver
from .drivers.video_driver import VideoDriver
from .drivers.calcium_imaging_driver import CalciumImagingDriver
from .drivers.motion_sensor_driver import MotionSensorDriver
from .drivers.ecog_driver import ECoGDriver
from .drivers.emg_driver import EMGDriver
from .drivers.eog_driver import EOGDriver
from .models import (
    EEGNetModel,
    CNNModel,
    RandomForestModel,
    SVMModel,
    KNNModel,
    GBDTModel,
    SimpleClassifier,
    TransformerModel,
    DinoV3Model,
)

from .agents import (
    VideoAgent,
    PoseAgent,
    FacialAgent,
    BlinkAgent,
    MotionAgent,
    CalciumAgent,
)
from .drivers.dataset_driver import DatasetDriver


def generate_pipeline_for_task(
    task_description: str,
    *,
    use_brainflow: bool = False,
    model_name: Optional[str] = None,
    fs: float = 250.0,
    channels: int = 8,
) -> Pipeline:
    """Generate a neurOS pipeline based on a task description.

    Parameters
    ----------
    task_description : str
        Natural language description of the intended BCI task (e.g. "2‑class
        motor imagery" or "SSVEP speller").
    use_brainflow : bool, optional
        Whether to instantiate a BrainFlow driver.  If False (default) a
        MockDriver is used.  If BrainFlow is not installed, the driver will
        fall back to a mock driver regardless of this flag.
    model_name : str, optional
        Explicit model name to override heuristic selection.  Must be one
        of "eegnet", "cnn", "random_forest", "svm", "knn", "gbdt" or
        "simple".
    fs : float, optional
        Sampling rate in Hz.  Defaults to 250.
    channels : int, optional
        Number of channels.  Defaults to 8.

    Returns
    -------
    Pipeline
        Configured pipeline ready for training and execution.
    """
    desc = task_description.lower()
    # Choose model and processing modality
    model: object
    if model_name:
        name = model_name.lower()
    else:
        name = ""
    # default to random forest; override with heuristics when no explicit name provided
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
    # instantiate model
    if name == "eegnet":
        model = EEGNetModel()
    elif name == "cnn":
        model = CNNModel()
    elif name == "transformer":
        model = TransformerModel()
    elif name == "dino" or name == "dino_v3":
        # use DinoV3Model for self‑supervised vision tasks
        model = DinoV3Model()
    elif name == "random_forest":
        model = RandomForestModel()
    elif name == "svm":
        model = SVMModel()
    elif name == "knn":
        model = KNNModel()
    elif name == "gbdt":
        model = GBDTModel()
    elif name == "simple":
        model = SimpleClassifier()
    else:
        raise ValueError(f"Unknown model name: {name}")

    # Choose bands based on task: narrow when expecting specific EEG frequency components
    bands: Optional[dict[str, tuple[float, float]]] = None
    if "ssvep" in desc:
        bands = {"alpha_beta": (8.0, 20.0)}
    elif "motor" in desc:
        bands = {"mu_beta": (8.0, 30.0)}
    # else leave None

    # determine modality and appropriate driver/processing agent
    processing_agent_class = None
    processing_kwargs: dict = {}
    # dataset reprocessing tasks
    # detect explicit mention of dataset or specific dataset names
    dataset_names = ["iris", "digits", "wine", "breast cancer", "breast_cancer", "cancer"]
    if any(k in desc for k in ["dataset", "data set", "reprocess", "reanalysis"]):
        # choose default dataset based on keywords in description
        selected = None
        for candidate in dataset_names:
            if candidate in desc:
                selected = candidate
                break
        # normalise dataset name
        if selected is None:
            selected = "iris"
        # map "breast cancer" to "breast_cancer"
        if selected in ("breast cancer", "cancer"):
            selected_name = "breast_cancer"
        else:
            selected_name = selected
        # instantiate dataset driver at the provided sampling rate
        driver = DatasetDriver(dataset_name=selected_name, sampling_rate=fs)
        # pass through raw features to model; use MotionAgent to forward samples unchanged
        from .agents import MotionAgent  # inline import to avoid circular dependencies
        processing_agent_class = MotionAgent
        processing_kwargs = {}
        # dataset tasks often benefit from strong models; if no explicit model requested,
        # choose RandomForest for tabular data
        if not model_name and name == "random_forest":
            model = RandomForestModel()
    # calcium imaging tasks
    elif any(k in desc for k in ["calcium", "imaging", "optical", "fluorescence", "two-photon"]):
        # Calcium imaging: treat fs as frame rate (frames per second)
        driver = CalciumImagingDriver(frame_rate=fs)
        # Use CalciumAgent to compute simple summary features (mean/std)
        processing_agent_class = CalciumAgent
        # imaging tasks benefit from convolutional or transformer models
        if not model_name and name in ["random_forest", "knn", "svm"]:
            # select CNN as default deep model for calcium images
            model = CNNModel()
    # video modality tasks (handle words like 'video', 'pose', 'facial', 'blink', 'face').  To avoid
    # misclassifying 'motor imagery' as an image task, we check for motor imagery before this branch.
    elif any(k in desc for k in ["video", "pose", "facial", "blink", "face"]):
        driver = VideoDriver(frame_rate=fs, resolution=(64, 64), channels=3)
        if "pose" in desc:
            processing_agent_class = PoseAgent
        elif any(k in desc for k in ["facial", "face"]):
            processing_agent_class = FacialAgent
        elif "blink" in desc:
            processing_agent_class = BlinkAgent
            # pass number of bands if specified (optional; default 4)
            processing_kwargs = {}
        else:
            # generic video processing: use VideoAgent
            processing_agent_class = VideoAgent
    # motion sensor tasks
    elif any(k in desc for k in ["motion", "imu", "movement", "acceleration"]):
        driver = MotionSensorDriver(sampling_rate=fs)
        processing_agent_class = MotionAgent
    # ECoG tasks
    elif any(k in desc for k in ["ecog", "corticography", "cortical", "intracranial"]):
        # ECoG tends to have high bandwidth and many channels; adjust sampling
        driver = ECoGDriver(sampling_rate=fs if fs else 1000.0, channels=channels if channels else 32)
        # default model for ECoG tasks: CNN or user‑selected
    # EMG tasks
    elif any(k in desc for k in ["emg", "muscle", "myography"]):
        driver = EMGDriver(sampling_rate=fs if fs else 500.0, channels=channels if channels else 8)
        # default model: RandomForest for EMG if not explicit
        if not model_name and name == "random_forest":
            # EMG tasks often work well with simple classifiers
            name = "random_forest"
    # EOG tasks
    elif any(k in desc for k in ["eog", "ocular", "eye"]):
        driver = EOGDriver(sampling_rate=fs if fs else 250.0, channels=channels if channels else 4)
        # default model: KNN for eye blink detection
        if not model_name and name == "random_forest":
            name = "knn"
    else:
        # EEG tasks: use BrainFlow or Mock
        if use_brainflow:
            driver = BrainFlowDriver(board_id=0, sampling_rate=fs, channels=channels)
        else:
            driver = MockDriver(sampling_rate=fs, channels=channels)
    # adjust model based on driver type when a default model was selected
    # If the selected model is the default RandomForestModel (i.e. from heuristics),
    # switch to a more appropriate model for certain modalities
    if not model_name:
        if isinstance(driver, ECoGDriver) and isinstance(model, RandomForestModel):
            # Use a CNN for high‑density intracranial signals
            model = CNNModel()
        elif isinstance(driver, EOGDriver) and isinstance(model, RandomForestModel):
            # Use KNN for eye movement data
            model = KNNModel()
        # For EMGDriver default RandomForest is suitable; no change

    return Pipeline(
        driver=driver,
        model=model,
        fs=fs,
        bands=bands,
        adaptation=True,
        processing_agent_class=processing_agent_class,
        processing_kwargs=processing_kwargs,
    )