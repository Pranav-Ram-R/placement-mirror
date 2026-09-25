"""Where session data is stored: %LOCALAPPDATA%\PlacementMirror\sessions on Windows.

PLACEMENT_MIRROR_DATA overrides the root folder (tests, automated runs). Session data
never goes in the repo.
"""

from __future__ import annotations

import os
from pathlib import Path


def data_root() -> Path:
    override = os.environ.get("PLACEMENT_MIRROR_DATA")
    if override:
        return Path(override)
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "PlacementMirror"


def sessions_dir() -> Path:
    return data_root() / "sessions"
