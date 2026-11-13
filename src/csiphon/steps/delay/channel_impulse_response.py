"""Channel impulse response: complex subcarrier CSI to the delay-domain CIR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import RealArray, SignalArray, as_signal_array
from csiphon.core.axes import Axis, AxisName
from csiphon.core.errors import LayoutError
from csiphon.core.layout import Layout
from csiphon.core.semantics import Representation, ValueKind
from csiphon.core.signal import Signal
from csiphon.pipeline.step import PointwiseStep
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming
from csiphon.steps.delay.delay_autocorrelation import (
    resolve_nfft,
    subcarrier_indices_of,
)


def _cir_estimator(subcarriers: RealArray, num_taps: int, nfft: int) -> SignalArray:
    """Matrix P (taps, subcarriers) that maps the CSI H to the CIR: `h = P @ H`.

    With the full number of taps, P is the inverse DFT restricted to the observed
    subcarriers. With fewer `num_taps`, it is a least-squares fit that assumes the
    CIR has only that many taps.
    """

    indices = subcarriers[:, None]
    taps = np.arange(num_taps, dtype=float)[None, :]

    # The forward model: the CSI that each delay tap produces on each subcarrier.
    forward = np.exp(-1j * 2.0 * np.pi * indices * taps / nfft) / np.sqrt(nfft)

    # Invert it to recover the taps from the observed CSI.
    return as_signal_array(np.linalg.pinv(forward))


@dataclass(frozen=True, slots=True)
class ChannelImpulseResponse(PointwiseStep):
    """Estimate the channel impulse response from the complex subcarrier CSI.

    Inverts the frequency response H[k] to the delay domain, handling missing
    subcarriers (a DC gap) correctly. With the default `num_taps=None` it returns
    the full CIR (one tap per DFT bin); set `num_taps` to fit a shorter,
    better-conditioned CIR when you know it is sparse. Pointwise in time, so it
    streams exactly.

    Works on a reduced subcarrier axis (e.g. after SelectAxis). If `nfft` is None,
    we infer it from the span of the subcarrier indices.
    """

    num_taps: int | None = field(
        default=None, metadata={"doc": "CIR length in taps (full DFT length if None)"}
    )
    nfft: int | None = field(
        default=None,
        metadata={"doc": "DFT size (inferred from the subcarrier indices if None)"},
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

    def _taps_and_nfft(self, indices: RealArray) -> tuple[int, int]:
        """Resolve (num_taps, nfft): the full DFT length when num_taps is None."""

        nfft = resolve_nfft(indices, self.nfft)
        return (nfft if self.num_taps is None else self.num_taps), nfft

    def output_layout(self, layout: Layout) -> Layout:
        """Replace the subcarrier axis with the estimated complex delay taps."""

        self.require_inputs(layout)
        # Fails at compile time if the layout does not say which subcarriers it has.
        indices = subcarrier_indices_of(layout)

        num_taps, nfft = self._taps_and_nfft(indices)
        if not 1 <= num_taps <= nfft:
            raise LayoutError(
                f"num_taps={num_taps} must be in [1, {nfft}] (at most nfft)."
            )
        delay = Axis.static(AxisName.DELAY, tuple(range(num_taps)), unit="tap")
        return layout.replace_axis(AxisName.SUBCARRIER, delay).with_representation(
            Representation.DELAY_RESPONSE
        )

    def transform_values(
        self, values: SignalArray, signal: Signal, out_layout: Layout
    ) -> SignalArray:
        """Estimate the CIR by a matmul with the pseudoinverse estimator matrix."""

        # Move the subcarriers to the last axis so the matmul acts on them (..., n_sub).
        position = signal.layout.axis_position(AxisName.SUBCARRIER)
        channel = np.moveaxis(as_signal_array(values), position, -1)

        subcarriers = subcarrier_indices_of(signal.layout)
        num_taps, nfft = self._taps_and_nfft(subcarriers)

        # The estimator maps subcarriers to delay taps, so this matmul
        # turns the CSI into the estimated CIR.
        estimator = _cir_estimator(subcarriers, num_taps, nfft)
        cir = channel @ estimator.T
        return as_signal_array(np.moveaxis(cir, -1, position))
