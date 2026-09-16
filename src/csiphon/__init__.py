"""csiphon: an online-first, schema-validated CSI preprocessing pipeline.

Build an immutable Pipeline from steps (branch and merge as needed), compile it
against an AcquisitionProfile into a Siphon, then run it over a whole recording
(`siphon.pour()`, or `siphon.measure()` to also learn what each step cost) or
chunk by chunk (`siphon.stream()`). A run returns Outlets, the named results.
"""

from __future__ import annotations

from csiphon.core import (
    AcquisitionProfile,
    Axis,
    AxisName,
    Layout,
    Representation,
    Signal,
    ValueKind,
    create_signal,
)
from csiphon.inspection import describe
from csiphon.pipeline import (
    Alignment,
    Concatenate,
    Exact,
    Fuse,
    GroupCost,
    Hold,
    Junction,
    Mean,
    MeasuredRun,
    MergeStrategy,
    Node,
    Outlets,
    Pipeline,
    PointwiseStep,
    Sequence,
    Siphon,
    Stack,
    Step,
    StepCost,
    Stream,
    StreamOperator,
    Sum,
    Time,
)
from csiphon.spec import Category, Description, StepSpec, Streaming

__all__ = [
    "AcquisitionProfile",
    "Alignment",
    "Axis",
    "AxisName",
    "Category",
    "Concatenate",
    "Description",
    "Exact",
    "Fuse",
    "GroupCost",
    "Hold",
    "Junction",
    "Layout",
    "Mean",
    "MeasuredRun",
    "MergeStrategy",
    "Node",
    "Outlets",
    "Pipeline",
    "PointwiseStep",
    "Representation",
    "Sequence",
    "Signal",
    "Siphon",
    "Stack",
    "Step",
    "StepCost",
    "StepSpec",
    "Stream",
    "StreamOperator",
    "Streaming",
    "Sum",
    "Time",
    "ValueKind",
    "create_signal",
    "describe",
]
