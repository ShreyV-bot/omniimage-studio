"""Plugin discovery.

Plugins are ordinary Python modules that register nodes with ``@node(..., source="plugin")``.
Discovery order:
  1. entry points in the group ``omniimage.nodes`` (pip-installable plugins)
  2. every ``*.py`` file in the directories listed in ``$OMNIIMAGE_PLUGIN_PATH`` and in ``~/.omniimage/plugins``

SECURITY: a plugin is arbitrary code running with your permissions. Only install plugins you trust.
Set ``OMNIIMAGE_NO_PLUGINS=1`` to disable discovery entirely.
"""
from __future__ import annotations
import importlib.util
import os
import sys
import warnings
from importlib import metadata
from pathlib import Path
from typing import List

_loaded: List[str] = []


def load_plugins() -> List[str]:
    if os.environ.get("OMNIIMAGE_NO_PLUGINS") == "1":
        return []
    try:
        eps = metadata.entry_points(group="omniimage.nodes")
    except Exception:
        eps = []
    for ep in eps:
        if ep.name in _loaded:
            continue
        try:
            ep.load()
            _loaded.append(ep.name)
        except Exception as e:  # a broken plugin must never break the app
            warnings.warn(f"omniimage plugin {ep.name!r} failed to load: {e}")
    dirs = [Path.home() / ".omniimage" / "plugins"]
    dirs += [Path(p) for p in os.environ.get("OMNIIMAGE_PLUGIN_PATH", "").split(os.pathsep) if p]
    for d in dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.py")):
            key = str(f.resolve())
            if key in _loaded:
                continue
            try:
                spec = importlib.util.spec_from_file_location(f"omniimage_plugin_{f.stem}", f)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                _loaded.append(key)
            except Exception as e:
                warnings.warn(f"omniimage plugin {f} failed to load: {e}")
    return list(_loaded)
