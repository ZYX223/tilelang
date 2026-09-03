"""Sunway target backend."""

from . import target as target  # Register target normalization first.
from . import backend as backend  # Register the backend manifest.
from . import op as op  # Register target-specific TileOp implementations.
