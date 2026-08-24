"""Core data model: axes, layout, signal, acquisition profile, sampling."""

from __future__ import annotations

from csiphon.core.arrays import (
    ComplexArray,
    RealArray,
    SignalArray,
    as_complex_array,
    as_real_array,
    as_signal_array,
)
from csiphon.core.axes import Axis, AxisName, Coordinate
from csiphon.core.errors import (
    ClogError,
    CompileError,
    DataError,
    LayoutError,
    MissingDependencyError,
    StreamingError,
)
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.sampling import (
    JitterWarning,
    check_jitter,
    effective_rate_hz,
    estimate_rate_hz,
    jitter_ratio,
)
from csiphon.core.semantics import Representation, ValueKind
from csiphon.core.signal import Signal, create_signal, empty_signal

__all__ = [
    "AcquisitionProfile",
    "Axis",
    "AxisName",
    "ClogError",
    "CompileError",
    "ComplexArray",
    "Coordinate",
    "DataError",
    "JitterWarning",
    "Layout",
    "LayoutError",
    "MissingDependencyError",
    "RealArray",
    "Representation",
    "Signal",
    "SignalArray",
    "StreamingError",
    "ValueKind",
    "as_complex_array",
    "as_real_array",
    "as_signal_array",
    "check_jitter",
    "create_signal",
    "effective_rate_hz",
    "empty_signal",
    "estimate_rate_hz",
    "jitter_ratio",
]
