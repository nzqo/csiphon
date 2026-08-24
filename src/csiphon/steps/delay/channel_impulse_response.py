"""Channel impulse response: complex subcarrier CSI to the delay-domain CIR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import RealArray, SignalArray, as_signal_array
from csiphon.core.axes import Axis, AxisName
from csiphon.core.errors import LayoutError
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.semantics import Representation, ValueKind
from csiphon.core.signal import Signal
from csiphon.pipeline.step import PointwiseStep
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming
from csiphon.steps.delay.delay_autocorrelation import resolve_nfft


def _cir_estimator(subcarriers: RealArray, num_taps: int, nfft: int) -> SignalArray:
    """Delay estimator P (taps, subcarriers) with `h = P @ H`, via the pseudoinverse.

    At the full DFT length P is the min-norm inverse, a plain inverse DFT of the
    observed tones; a shorter `num_taps` makes it an over-determined least-squares
    fit that assumes the CIR has only that many taps.
    """

    tones = subcarriers[:, None]
    taps = np.arange(num_taps, dtype=float)[None, :]

    # The forward model: how each delay tap shows up across the tones.
    forward = np.exp(-1j * 2.0 * np.pi * tones * taps / nfft) / np.sqrt(nfft)

    # Invert it to recover the taps from the observed tones.
    return as_signal_array(np.linalg.pinv(forward))


@dataclass(frozen=True, slots=True)
class ChannelImpulseResponse(PointwiseStep):
    """Estimate the channel impulse response from the complex subcarrier CSI.

    Inverts the observed complex tones to the delay domain, handling missing
    subcarriers (a DC gap) correctly. With the default `num_taps=None` it returns
    the full CIR (one tap per DFT bin); set `num_taps` to fit a shorter,
    better-conditioned CIR when you know it is sparse. Pointwise in time, so it
    streams exactly.
    """

    num_taps: int | None = field(
        default=None, metadata={"doc": "CIR length in taps (full DFT length if None)"}
    )
    nfft: int | None = field(
        default=None,
        metadata={"doc": "DFT size (inferred from subcarrier count if None)"},
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="channel-impulse-response",
        summary="estimate the complex CIR by least squares over active subcarriers",
        category=Category.DELAY,
        admissible_values=(ValueKind.COMPLEX,),
        admissible_reprs=None,
        requires_axes=(AxisName.SUBCARRIER,),
        layout_effect=LayoutEffect(
            replaces=((AxisName.SUBCARRIER, AxisName.DELAY),),
            note="delay-domain response",
        ),
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def _taps_and_nfft(self, profile: AcquisitionProfile) -> tuple[int, int]:
        """Resolve (num_taps, nfft): the full DFT length when num_taps is None."""

        nfft = resolve_nfft(profile.n_subcarriers, self.nfft)
        return (nfft if self.num_taps is None else self.num_taps), nfft

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Replace the subcarrier axis with the estimated complex delay taps."""

        self.require_inputs(layout)
        layout.require_static_axis(AxisName.SUBCARRIER)
        num_taps, nfft = self._taps_and_nfft(profile)
        if not 1 <= num_taps <= nfft:
            raise LayoutError(
                f"num_taps={num_taps} must be in [1, {nfft}] (at most nfft)."
            )
        delay = Axis.static(AxisName.DELAY, tuple(range(num_taps)), unit="tap")
        return layout.replace_axis(AxisName.SUBCARRIER, delay).with_representation(
            Representation.DELAY_RESPONSE
        )

    def transform_values(
        self,
        values: SignalArray,
        signal: Signal,
        out_layout: Layout,
        profile: AcquisitionProfile,
    ) -> SignalArray:
        """Estimate the CIR by a matmul with the pseudoinverse estimator matrix."""

        # Move the subcarriers to the last axis so the matmul acts on them (..., n_sub).
        position = signal.layout.axis_position(AxisName.SUBCARRIER)
        channel = np.moveaxis(as_signal_array(values), position, -1)

        num_taps, nfft = self._taps_and_nfft(profile)
        subcarriers = np.asarray(profile.subcarrier_indices, dtype=float)
        # The estimator maps subcarriers to delay taps, so this matmul
        # turns the CSI's tones into the estimated CIR.
        estimator = _cir_estimator(subcarriers, num_taps, nfft)
        cir = channel @ estimator.T
        return as_signal_array(np.moveaxis(cir, -1, position))
