"""OmniImage: an open-source, node-based, scriptable media engine (image, video frames, audio, LUTs, VFX, SFX)."""
import os as _os
_os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")   # must be set before cv2 is imported

__version__ = "0.8.0"

from .core import Graph, Project, Ctx, GraphError, ParamError, PortType, AudioBuf, REGISTRY, list_specs, get_spec  # noqa: E402
from . import nodes as _nodes  # noqa: E402,F401  (registers built-in nodes)
from .lut import LUT3D, bake_lut  # noqa: E402
from .plugins import load_plugins  # noqa: E402
from .timeline import (Timeline, Track, Clip, AudioClip, Transition, Bus, Duck, Master,  # noqa: E402
                       render_timeline)
from .loudness import measure_lufs, normalize_lufs  # noqa: E402

load_plugins()
