"""Pipeline construction, compilation, and execution."""

# The re-export __all__ here overlaps the top-level package's by design.
# pylint: disable=duplicate-code
from __future__ import annotations

from csiphon.pipeline._align_ops import align_on_time
from csiphon.pipeline.merges import (
    Alignment,
    Concatenate,
    Exact,
    Fuse,
    Hold,
    Junction,
    Mean,
    MergeStrategy,
    Stack,
    Sum,
    Time,
)
from csiphon.pipeline.pipeline import Node, Pipeline, Siphon
from csiphon.pipeline.runners import Outlets, Stream, concat_signals
from csiphon.pipeline.sequence import Sequence
from csiphon.pipeline.step import PointwiseStep, Step, StreamOperator

__all__ = [
    "Alignment",
    "Concatenate",
    "Exact",
    "Fuse",
    "Hold",
    "Junction",
    "Mean",
    "MergeStrategy",
    "Node",
    "Outlets",
    "Pipeline",
    "PointwiseStep",
    "Sequence",
    "Siphon",
    "Stack",
    "Step",
    "Stream",
    "StreamOperator",
    "Sum",
    "Time",
    "align_on_time",
    "concat_signals",
]
