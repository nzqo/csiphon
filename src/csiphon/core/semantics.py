"""Runtime descriptions of what a signal and its values represent.

These two orthogonal enums let the compiler reject physically meaningless
pipelines (for example, taking the decibels of an already-complex tensor)
before any numerical work runs.
"""

from enum import StrEnum


class Representation(StrEnum):
    """The underlying signal the tensor expresses (its physical domain)."""

    # fmt: off
    CHANNEL_FREQUENCY_RESPONSE = "channel frequency response"
    CROSS_SPECTRUM             = "cross-spectrum"
    RATIO                      = "ratio"
    DELAY_RESPONSE             = "delay-domain response"
    TIME_FREQUENCY             = "time-frequency representation"
    WAVELET_COEFFICIENTS       = "wavelet coefficients"
    AUTOCORRELATION            = "autocorrelation"
    FEATURE_VECTOR             = "feature vector"
    # fmt: on

    @property
    def short(self) -> str:
        """A compact label for tables; the full value is used in verbose views."""

        return _REPRESENTATION_SHORT.get(self, self.value)


class ValueKind(StrEnum):
    """The form that signal's values take (magnitude, power, complex, ...)."""

    # fmt: off
    COMPLEX       = "complex"
    REAL          = "real-valued"
    MAGNITUDE     = "magnitude"
    POWER         = "power"
    PHASE_RADIANS = "phase in radians"
    DECIBELS      = "decibels"
    # fmt: on

    @property
    def short(self) -> str:
        """A compact label for tables; the full value is used in verbose views."""

        return _VALUE_KIND_SHORT.get(self, self.value)


# Compact labels for the concise pipeline table (full values shown elsewhere).
# fmt: off
_REPRESENTATION_SHORT: dict[Representation, str] = {
    Representation.CHANNEL_FREQUENCY_RESPONSE : "CFR",
    Representation.CROSS_SPECTRUM             : "cross-spec",
    Representation.RATIO                      : "ratio",
    Representation.DELAY_RESPONSE             : "delay",
    Representation.TIME_FREQUENCY             : "time-freq",
    Representation.WAVELET_COEFFICIENTS       : "wavelet",
    Representation.AUTOCORRELATION            : "autocorr",
    Representation.FEATURE_VECTOR             : "features",
}

_VALUE_KIND_SHORT: dict[ValueKind, str] = {
    ValueKind.COMPLEX       : "complex",
    ValueKind.REAL          : "real",
    ValueKind.MAGNITUDE     : "magnitude",
    ValueKind.POWER         : "power",
    ValueKind.PHASE_RADIANS : "phase",
    ValueKind.DECIBELS      : "dB",
}
# fmt: on
