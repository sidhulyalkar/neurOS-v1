"""
Agents for neurOS.

Agents encapsulate different functional units of a BCI pipeline.  Each
concrete agent inherits from :class:`BaseAgent` and implements a :meth:`run`
coroutine that performs its task.  The orchestrator coordinates agents to
build a complete processing chain.
"""

from neuros.agents.base_agent import BaseAgent  # noqa: F401
from neuros.agents.device_agent import DeviceAgent  # noqa: F401
from neuros.agents.processing_agent import ProcessingAgent  # noqa: F401
from neuros.agents.model_agent import ModelAgent  # noqa: F401
from neuros.agents.orchestrator_agent import Orchestrator  # noqa: F401

# multimodal agents
from neuros.agents.video_agent import VideoAgent  # noqa: F401
from neuros.agents.pose_agent import PoseAgent  # noqa: F401
from neuros.agents.facial_agent import FacialAgent  # noqa: F401
from neuros.agents.blink_agent import BlinkAgent  # noqa: F401
from neuros.agents.motion_agent import MotionAgent  # noqa: F401
from neuros.agents.calcium_agent import CalciumAgent  # noqa: F401
# new multimodal agents
from neuros.agents.fusion_agent import FusionAgent  # noqa: F401
from neuros.agents.multimodal_orchestrator import MultiModalOrchestrator  # noqa: F401
# NotebookAgent and ModalityManagerAgent provide higher‑level automation and
# are intentionally not imported here to avoid circular dependencies with
# ``neuros.autoconfig`` and ``neuros.pipeline``.  Import them explicitly
# where needed (e.g. in the CLI or user code).
