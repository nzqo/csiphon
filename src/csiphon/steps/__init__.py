"""The step library, grouped into categories.

Each step lives in its own file under a category subpackage (components, scaling,
cleaning, calibration, normalization, baseline, filtering, temporal_features,
delay, time_frequency, pooling, statistics, reduction, restructuring,
resampling). Everything is re-exported here, so `from csiphon.steps import
Magnitude` works no matter which file a step lives in. Call
`csiphon.describe(step)` to see any step's contract.

Steps that need an optional dependency (SynchrosqueezedPower needs ssqueezepy,
ButterworthFilter needs scipy) import it lazily, so the base install stays
numpy-only.
"""

from csiphon.steps.baseline import RunningMeanSubtract, TemporalMeanSubtract
from csiphon.steps.calibration import AxisReference, LinearPhaseCorrection
from csiphon.steps.cleaning import NanScrub, NoiseFloorClip
from csiphon.steps.components import (
    ImagPart,
    Magnitude,
    Phase,
    Power,
    RealPart,
    UnitPhase,
)
from csiphon.steps.delay import (
    ChannelImpulseResponse,
    DelayAutocorrelation,
    resolve_nfft,
)
from csiphon.steps.filtering import ButterworthFilter, SavitzkyGolay
from csiphon.steps.normalization import (
    GainNormalize,
    GlobalMaxNormalize,
    PerFrameMaxNormalize,
)
from csiphon.steps.pooling import (
    DyadicFrequencyBands,
    FixedSizeWindowSum,
    MeanOverAxes,
)
from csiphon.steps.reduction import (
    DelayTaps,
    PCABasis,
    PrincipalComponents,
    RobustPca,
    SelectAxis,
    fit_pca_basis,
)
from csiphon.steps.resampling import (
    Bursty,
    CubicSpline,
    DropSamples,
    FillMethod,
    Hold,
    Independent,
    Linear,
    LossModel,
    Nearest,
    PolarLinear,
    Resample,
    SubsampleEvery,
)
from csiphon.steps.restructuring import FoldAxes, StackAmpPhase
from csiphon.steps.scaling import LogScale, ToDecibels
from csiphon.steps.statistics import CovarianceSpectrum, LocalPcaBias
from csiphon.steps.temporal_features import (
    TimeDifference,
    WindowedSlope,
    WindowedVariance,
)
from csiphon.steps.time_frequency import (
    ComplexStftMagnitude,
    Multitaper,
    SynchrosqueezedPower,
    WindowedFFTPower,
)

__all__ = [
    # ----------------------
    # Steps
    "AxisReference",
    "Bursty",
    "ButterworthFilter",
    "ChannelImpulseResponse",
    "ComplexStftMagnitude",
    "CovarianceSpectrum",
    "CubicSpline",
    "DelayAutocorrelation",
    "DelayTaps",
    "DropSamples",
    "DyadicFrequencyBands",
    "FillMethod",
    "FixedSizeWindowSum",
    "FoldAxes",
    "GainNormalize",
    "GlobalMaxNormalize",
    "Hold",
    "ImagPart",
    "Independent",
    "Linear",
    "LinearPhaseCorrection",
    "LocalPcaBias",
    "LogScale",
    "LossModel",
    "Magnitude",
    "MeanOverAxes",
    "Multitaper",
    "NanScrub",
    "Nearest",
    "NoiseFloorClip",
    "PCABasis",
    "PerFrameMaxNormalize",
    "Phase",
    "PolarLinear",
    "Power",
    "PrincipalComponents",
    "RealPart",
    "Resample",
    "RobustPca",
    "RunningMeanSubtract",
    "SavitzkyGolay",
    "SelectAxis",
    "StackAmpPhase",
    "SubsampleEvery",
    "SynchrosqueezedPower",
    "TemporalMeanSubtract",
    "TimeDifference",
    "ToDecibels",
    "UnitPhase",
    "WindowedFFTPower",
    "WindowedSlope",
    "WindowedVariance",
    # ----------------------
    # helper functions
    "fit_pca_basis",
    "resolve_nfft",
]
