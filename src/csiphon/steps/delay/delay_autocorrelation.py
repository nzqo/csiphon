"""Delay-domain autocorrelation from the subcarrier power (inverse DFT of |H|^2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import SignalArray, as_signal_array
from csiphon.core.axes import Axis, AxisName
from csiphon.core.errors import LayoutError
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.semantics import Representation, ValueKind
from csiphon.core.signal import Signal
from csiphon.pipeline.step import PointwiseStep
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming

# WiFi uses a fixed FFT size (a power of two) per bandwidth, and only some of its
# bins carry data. Given the number of used subcarriers, pick the smallest of
# these FFT sizes that still fits them: 64 (20 MHz), 128 (40 MHz, ~114 used),
# 256 (80 MHz), 512 (160 MHz).
_FFT_SIZES = (64, 128, 256, 512)


def resolve_nfft(n_subcarriers: int, nfft: int | None = None) -> int:
    """Return nfft, inferring it from the subcarrier count when you leave it None."""

    if nfft is not None:
        if nfft < 1:
            raise LayoutError(f"nfft must be >= 1, got {nfft}.")
        return nfft
    for size in _FFT_SIZES:
        if n_subcarriers <= size:
            return size
    raise LayoutError(
        f"cannot infer nfft for {n_subcarriers} subcarriers "
        "(>512); pass nfft explicitly."
    )


@dataclass(frozen=True, slots=True)
class DelayAutocorrelation(PointwiseStep):
    """Delay-domain autocorrelation of the channel, from the subcarrier power.

    It squares the input to power (one value per subcarrier), then takes the
    inverse DFT of that power across the subcarriers. By the Wiener-Khinchin
    relation, the inverse DFT of the power spectrum is the autocorrelation of the
    channel impulse response. So the output is that autocorrelation, one complex
    value per delay tap. By default that is every tap (0 .. nfft-1); `first_tap`
    and `num_taps` restrict it to a range, which costs only that range.

    The basis uses the physical subcarrier indices, so this is tied to the
    subcarrier axis (not axis-generic). To get the actual impulse response
    instead of its autocorrelation, use ChannelImpulseResponse. To turn the taps
    into real features, follow with DelayTaps.
    """

    nfft: int | None = field(
        default=None,
        metadata={"doc": "DFT size (inferred from subcarrier count if None)"},
    )
    first_tap: int = field(default=0, metadata={"doc": "first delay tap kept"})
    num_taps: int | None = field(
        default=None,
        metadata={"doc": "taps kept from first_tap on (every remaining tap if None)"},
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="delay-autocorrelation",
        summary="delay-domain autocorrelation from the subcarrier power",
        category=Category.DELAY,
        admissible_values=(ValueKind.MAGNITUDE, ValueKind.REAL),
        admissible_reprs=None,
        requires_axes=(AxisName.SUBCARRIER,),
        layout_effect=LayoutEffect(
            replaces=((AxisName.SUBCARRIER, AxisName.DELAY),),
            value_kind=ValueKind.COMPLEX,
            note="delay autocorrelation",
        ),
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def kept_taps(self, nfft: int) -> range:
        """The delay taps the output carries, checked to lie within nfft."""

        if self.first_tap < 0:
            raise LayoutError(f"first_tap must be >= 0, got {self.first_tap}.")
        if self.num_taps is not None and self.num_taps < 1:
            raise LayoutError(f"num_taps must be >= 1, got {self.num_taps}.")
        last_tap = nfft if self.num_taps is None else self.first_tap + self.num_taps
        if self.first_tap >= nfft or last_tap > nfft:
            raise LayoutError(
                f"taps {self.first_tap}..{last_tap - 1} exceed nfft {nfft}."
            )
        return range(self.first_tap, last_tap)

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Replace the subcarrier axis with the kept delay taps."""

        self.require_inputs(layout)
        layout.require_static_axis(AxisName.SUBCARRIER)

        nfft = resolve_nfft(profile.n_subcarriers, self.nfft)
        taps = self.kept_taps(nfft)
        delay = Axis.static(AxisName.DELAY, tuple(taps), unit="tap")
        return (
            layout.replace_axis(AxisName.SUBCARRIER, delay)
            .with_representation(Representation.AUTOCORRELATION)
            .with_values(ValueKind.COMPLEX)
        )

    def transform_values(
        self,
        values: SignalArray,
        signal: Signal,
        out_layout: Layout,
        profile: AcquisitionProfile,
    ) -> SignalArray:
        """Square the input to power, then inverse-DFT it across the subcarriers."""

        # Put subcarriers last so the projection onto delay taps is a plain matmul.
        position = signal.layout.axis_position(AxisName.SUBCARRIER)
        amplitude = np.moveaxis(values, position, -1)

        # kernel[k, t] = exp(2j*pi*k*t/nfft): the inverse-DFT basis from subcarrier
        # index k to delay tap t, for the kept taps only.
        subcarriers = np.asarray(profile.subcarrier_indices, dtype=float)
        nfft = resolve_nfft(profile.n_subcarriers, self.nfft)
        taps = np.asarray(self.kept_taps(nfft))
        kernel = np.exp(1j * 2.0 * np.pi * np.outer(subcarriers, taps) / nfft)

        # Inverse DFT of the power |H|^2 across subcarriers, divided by their count
        # so it is a mean. By Wiener-Khinchin, this is the autocorrelation. The
        # power is flattened to one (rows, subcarriers) matrix and multiplied by
        # the real and the imaginary part of the kernel separately: two real
        # GEMMs over one large matrix run several times faster than a complex
        # GEMM batched over the leading axes, for the same result.
        power = (np.abs(amplitude) ** 2).reshape(-1, amplitude.shape[-1])
        coefficients = power @ kernel.real + 1j * (power @ kernel.imag)
        coefficients = coefficients.reshape(*amplitude.shape[:-1], taps.size)
        coefficients /= amplitude.shape[-1]
        return as_signal_array(np.moveaxis(coefficients, -1, position))
