"""
Agents for neurOS.

Agents encapsulate different functional units of a BCI pipeline.  Each
concrete agent inherits from :class:`BaseAgent` and implements a :meth:`run`
coroutine that performs its task.  The orchestrator coordinates agents to
build a complete processing chain.
"""

from .base_agent import BaseAgent  # noqa: F401
from .device_agent import DeviceAgent  # noqa: F401
from .processing_agent import ProcessingAgent  # noqa: F401
from .model_agent import ModelAgent  # noqa: F401
from .orchestrator_agent import Orchestrator  # noqa: F401

# multimodal agents
from .video_agent import VideoAgent  # noqa: F401
from .pose_agent import PoseAgent  # noqa: F401
from .facial_agent import FacialAgent  # noqa: F401
from .blink_agent import BlinkAgent  # noqa: F401
from .motion_agent import MotionAgent  # noqa: F401
from .calcium_agent import CalciumAgent  # noqa: F401
# new multimodal agents
from .fusion_agent import FusionAgent  # noqa: F401
from .multimodal_orchestrator import MultiModalOrchestrator  # noqa: F401
# NotebookAgent and ModalityManagerAgent provide higher‑level automation and
# are intentionally not imported here to avoid circular dependencies with
# ``neuros.autoconfig`` and ``neuros.pipeline``.  Import them explicitly
# where needed (e.g. in the CLI or user code).