"""Delay-domain autocorrelation from the subcarrier power (inverse DFT of |H|^2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import RealArray, SignalArray, as_real_array, as_signal_array
from csiphon.core.axes import Axis, AxisName
from csiphon.core.errors import LayoutError
from csiphon.core.layout import Layout
from csiphon.core.semantics import Representation, ValueKind
from csiphon.core.signal import Signal
from csiphon.pipeline.step import PointwiseStep
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming

# WiFi uses a fixed FFT size (a power of two) per bandwidth, and the used
# subcarriers sit at indices around zero on that grid: 64 (20 MHz), 128 (40 MHz,
# indices -58 to 58), 256 (80 MHz), 512 (160 MHz).
_FFT_SIZES = (64, 128, 256, 512)


def resolve_nfft(subcarrier_indices: RealArray, nfft: int | None = None) -> int:
    """Return nfft, inferring it from the span of the subcarrier indices if None.

    The smallest FFT size larger than the span keeps every index on its own bin.

    Note: nfft is decided solely by the span of the subcarrier indices, not by
    their spacing. With different spacing, the axis unit would change (e.g. one
    bin would be 12.5 ns vs 50 ns under the narrower HE spacing). The calculation
    is still the same though. Currently, output axis of this step are in units of
    "tap", so we can ignore the spacing for now. If we wanted it in physical SI
    units, we will need to propagate it through the axis.
    """

    if nfft is not None:
        if nfft < 1:
            raise LayoutError(f"nfft must be >= 1, got {nfft}.")
        return nfft

    span = float(subcarrier_indices.max() - subcarrier_indices.min())
    for size in _FFT_SIZES:
        if span < size:
            return size
    raise LayoutError(
        f"cannot infer nfft for subcarrier indices spanning {span:.0f} "
        "(>= 512); pass nfft explicitly."
    )


def subcarrier_indices_of(layout: Layout) -> RealArray:
    """Return the physical subcarrier indices of the layout's subcarrier axis.

    They are the axis coordinates, so a SelectAxis upstream is respected.
    """

    axis = layout.require_static_axis(AxisName.SUBCARRIER)
    if axis.coordinates is None:
        raise LayoutError(
            "The subcarrier axis has no coordinates, so we cannot tell which "
            "subcarrier indices the data carries. Give the axis the indices as "
            "its coordinates."
        )
    return as_real_array(np.asarray(axis.coordinates, dtype=float))


@dataclass(frozen=True, slots=True)
class DelayAutocorrelation(PointwiseStep):
    """Autocorrelation of the channel impulse response, from the subcarrier power.

    We square the input to get the power per subcarrier, then take the inverse DFT
    of that power over the subcarriers. By the Wiener-Khinchin relation, the result
    is the autocorrelation of the channel impulse response: one complex value per
    delay tap. By default we return all taps (0 to nfft-1). If `first_tap` or
    `num_taps` is set, we compute only that range of taps.

    The DFT uses the subcarrier indices of the input axis, so this step only works
    on the subcarrier axis. That axis may have been reduced earlier (e.g. by
    SelectAxis). If `nfft` is None, we infer it from the span of the subcarrier
    indices.

    The output is complex. For real values, follow with Magnitude, RealPart or
    ImagPart. For the impulse response itself, use ChannelImpulseResponse.
    """

    nfft: int | None = field(
        default=None,
        metadata={"doc": "DFT size (inferred from the subcarrier indices if None)"},
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

    def output_layout(self, layout: Layout) -> Layout:
        """Replace the subcarrier axis with the kept delay taps."""

        self.require_inputs(layout)

        # Fails at compile time if the layout does not say which subcarriers it has.
        indices = subcarrier_indices_of(layout)
        nfft = resolve_nfft(indices, self.nfft)
        taps = self.kept_taps(nfft)
        delay = Axis.static(AxisName.DELAY, tuple(taps), unit="tap")
        return (
            layout.replace_axis(AxisName.SUBCARRIER, delay)
            .with_representation(Representation.AUTOCORRELATION)
            .with_values(ValueKind.COMPLEX)
        )

    def transform_values(
        self, values: SignalArray, signal: Signal, out_layout: Layout
    ) -> SignalArray:
        """Square the input to power, then inverse-DFT it across the subcarriers."""

        # Put subcarriers last so the projection onto delay taps is a plain matmul.
        position = signal.layout.axis_position(AxisName.SUBCARRIER)
        amplitude = np.moveaxis(values, position, -1)

        # kernel[k, t] = exp(2j*pi*k*t/nfft): the inverse-DFT basis from subcarrier
        # index k to delay tap t, for the kept taps only.
        subcarriers = subcarrier_indices_of(signal.layout)
        nfft = resolve_nfft(subcarriers, self.nfft)
        taps = np.asarray(self.kept_taps(nfft))
        kernel = np.exp(1j * 2.0 * np.pi * np.outer(subcarriers, taps) / nfft)

        # Inverse DFT of the power |H|^2 across subcarriers, divided by their count
        # so it is a mean. By Wiener-Khinchin, this is the autocorrelation.
        #
        # The input is real (a magnitude or real values), so the power is its
        # square. It is flattened to one (rows, subcarriers) matrix. The real and
        # the imaginary part of the kernel sit side by side, so a single real
        # matrix product gives both parts of every tap and reads the big matrix
        # once. This runs several times faster than a complex product batched over
        # the leading axes. Dividing the small kernel by the subcarrier count here
        # makes the product a mean.
        power = np.square(amplitude).reshape(-1, amplitude.shape[-1])
        kernel = kernel / amplitude.shape[-1]
        real_and_imag = power @ np.concatenate([kernel.real, kernel.imag], axis=1)
        coefficients = (
            real_and_imag[:, : taps.size] + 1j * real_and_imag[:, taps.size :]
        )
        coefficients = coefficients.reshape(*amplitude.shape[:-1], taps.size)
        return as_signal_array(np.moveaxis(coefficients, -1, position))
