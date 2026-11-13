"""SHARP's phase cleaning (Meneghello et al., IEEE TMC 2023) as a step."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import (
    ComplexArray,
    RealArray,
    SignalArray,
    as_complex_array,
    as_real_array,
    as_signal_array,
)
from csiphon.core.axes import AxisName
from csiphon.core.errors import LayoutError
from csiphon.core.layout import Layout
from csiphon.core.semantics import ValueKind
from csiphon.core.signal import Signal
from csiphon.pipeline.step import PointwiseStep
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming
from csiphon.steps.delay.delay_autocorrelation import subcarrier_indices_of

# The sparse fit makes arrays of (vectors, delays) numbers. Cleaning the vectors in
# blocks keeps that memory the same for a short and for a long recording.
_VECTORS_PER_BLOCK = 4096


def fit_sparse_paths(
    csi: ComplexArray, delay_basis: ComplexArray, l1_weight: float, iterations: int
) -> ComplexArray:
    """Fit a few paths to each row of `csi`, and return their complex amplitudes.

    A path is a delay with an amplitude. Column `d` of `delay_basis` is what a path
    with delay `d` looks like across the subcarriers, so a row of `csi` is
    approximated by `amplitudes @ delay_basis.T`. Few paths are wanted, so the
    amplitudes minimize

        |csi - amplitudes @ delay_basis.T|^2 / 2
        + l1_weight * (|real part|_1 + |imaginary part|_1)

    with FISTA (a gradient method with momentum). All rows are fitted at once.
    """

    # The gradient steps must not be longer than 1 / (the basis' largest gain)^2.
    step = 1.0 / np.linalg.norm(delay_basis, 2) ** 2

    amplitudes = np.zeros((csi.shape[0], delay_basis.shape[1]), dtype=np.complex128)
    lookahead = amplitudes
    momentum = 1.0

    for _ in range(iterations):
        # Move the lookahead point against the gradient of the squared error.
        error = lookahead @ delay_basis.T - csi
        stepped = lookahead - step * error @ delay_basis.conj()

        # The L1 penalty shrinks real and imaginary parts towards zero, each on its
        # own. Viewing the complex numbers as pairs of floats does both at once.
        parts = stepped.view(np.float64)
        shrunk = np.sign(parts) * np.maximum(np.abs(parts) - step * l1_weight, 0.0)
        updated = shrunk.view(np.complex128)

        # Momentum: continue from a point that overshoots the update a little.
        next_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum**2)) / 2.0
        overshoot = (momentum - 1.0) / next_momentum
        lookahead = updated + overshoot * (updated - amplitudes)
        amplitudes, momentum = updated, next_momentum

    return amplitudes


def _remove_linear_phase_trend(vectors: ComplexArray) -> ComplexArray:
    """Remove the linear trend of the phase across the subcarriers, per vector."""

    # Fit phase = offset + slope * position across the subcarriers, by least squares
    # on the unwrapped phase, and subtract that line.
    phase = np.unwrap(np.angle(vectors), axis=1)
    positions = np.arange(vectors.shape[1])
    line_basis = np.stack([np.ones(vectors.shape[1]), positions], axis=1)
    line_coefficients = np.linalg.lstsq(line_basis, phase.T, rcond=None)[0]
    flattened_phase = phase - (line_basis @ line_coefficients).T

    return as_complex_array(np.abs(vectors) * np.exp(1j * flattened_phase))


# One setting per tunable of the method justifies the attribute count.
@dataclass(frozen=True, slots=True)
class SparseFitSanitize(PointwiseStep):  # pylint: disable=too-many-instance-attributes
    """Clean the CSI phase of every packet, as SHARP does (Meneghello et al., 2023).

    The phase of raw CSI is dominated by offsets that change from packet to packet
    (carrier and sampling frequency offset, packet detection delay). They hide the
    phase changes that motion causes. All paths of a packet share these offsets, so
    they cancel when every path is given relative to the strongest one. For each
    packet, we:

    1. fit a few paths (a delay and an amplitude each) to the CSI, first on a coarse
       grid of delays, then on a fine grid around the strongest path,
    2. rebuild the CSI from the fitted paths, with every path's phase relative to
       the strongest path,
    3. remove the linear trend that the phase still has across the subcarriers.

    Every vector of subcarriers is cleaned on its own, so all other axes (time,
    antennas, ...) only decide which vector it is. The sparse fit expects the input
    magnitudes over their mean, so put a GainNormalize before this step. The scale of
    the output is arbitrary (rebuilding multiplies by the strongest path's
    amplitude), so put a GainNormalize after it too, as SHARP does.

    SHARP solves the sparse fit one packet at a time with OSQP. We solve the same
    objective with FISTA on many packets at once, which is faster but only
    approximate after a fixed number of `iterations`. SHARP's released code also
    applies the first packet's linear phase trend to all packets. We remove each
    packet's own trend. The two differ by a constant phase per subcarrier, which
    leaves the Doppler spectrum unchanged.
    """

    lasso_weight: float = field(
        default=0.1,
        metadata={"doc": "L1 penalty weight (fewer paths if larger)"},
    )
    iterations: int = field(
        default=25, metadata={"doc": "FISTA iterations per sparse fit"}
    )
    delay_range_s: tuple[float, float] = field(
        default=(-0.3e-6, 0.5e-6), metadata={"doc": "range of path delays, seconds"}
    )
    coarse_step_s: float = field(
        default=1e-7, metadata={"doc": "spacing of the coarse delay grid, seconds"}
    )
    fine_step_s: float = field(
        default=5e-9, metadata={"doc": "spacing of the fine delay grid, seconds"}
    )
    fine_before_s: float = field(
        default=2e-7,
        metadata={"doc": "fine grid starts this long before the strongest path"},
    )
    fine_after_s: float = field(
        default=2.5e-7,
        metadata={"doc": "fine grid ends this long after the strongest path"},
    )
    fit_every: int = field(
        default=2, metadata={"doc": "fit on every n-th subcarrier (rebuild on all)"}
    )
    subcarrier_spacing_hz: float = field(
        default=312.5e3,
        metadata={
            "doc": "frequency between subcarrier indices (802.11n/ac: 312.5 kHz)"
        },
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="sparse-fit-sanitize",
        summary="phase relative to the strongest path of a sparse multipath fit",
        category=Category.CALIBRATION,
        admissible_values=(ValueKind.COMPLEX,),
        admissible_reprs=None,
        requires_axes=(AxisName.SUBCARRIER,),
        layout_effect=LayoutEffect(note="phase relative to the strongest path"),
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def output_layout(self, layout: Layout) -> Layout:
        """The layout stays the same; only the values are cleaned."""

        self.require_inputs(layout)

        # Fails at compile time if the layout does not say which subcarriers it has.
        subcarrier_indices_of(layout)

        low, high = self.delay_range_s
        if low >= high:
            raise LayoutError(f"delay_range_s must be increasing, got {(low, high)}.")
        if self.coarse_step_s <= 0 or self.fine_step_s <= 0:
            raise LayoutError("coarse_step_s and fine_step_s must be > 0.")
        if self.fit_every < 1:
            raise LayoutError(f"fit_every must be >= 1, got {self.fit_every}.")
        if self.iterations < 1:
            raise LayoutError(f"iterations must be >= 1, got {self.iterations}.")
        return layout

    def transform_values(
        self, values: SignalArray, signal: Signal, out_layout: Layout
    ) -> SignalArray:
        """Clean every vector of subcarriers."""

        # Put the subcarriers last and list all other positions as one long list of
        # vectors, since each vector is cleaned on its own.
        position = signal.layout.axis_position(AxisName.SUBCARRIER)
        moved = np.moveaxis(as_complex_array(values), position, -1)
        vectors = moved.reshape(-1, moved.shape[-1])
        frequencies = subcarrier_indices_of(signal.layout) * self.subcarrier_spacing_hz

        cleaned = np.empty(vectors.shape, dtype=np.complex128)
        for start in range(0, len(vectors), _VECTORS_PER_BLOCK):
            block = slice(start, start + _VECTORS_PER_BLOCK)
            cleaned[block] = self._clean_vectors(vectors[block], frequencies)

        restored = np.moveaxis(cleaned.reshape(moved.shape), -1, position)
        return as_signal_array(restored)

    def _clean_vectors(
        self, vectors: ComplexArray, frequencies: RealArray
    ) -> ComplexArray:
        """Clean vectors of subcarriers (one per row)."""

        # Find where each vector's strongest path is, roughly.
        strongest_delays = self._coarse_strongest_delays(vectors, frequencies)

        # Refit around that path, and rebuild the vector relative to it.
        rebuilt = self._rebuild_relative_to_strongest(
            vectors, frequencies, strongest_delays
        )
        return _remove_linear_phase_trend(rebuilt)

    def _coarse_strongest_delays(
        self, vectors: ComplexArray, frequencies: RealArray
    ) -> RealArray:
        """Return, for each vector, the delay of its strongest path on a coarse grid."""

        low, high = self.delay_range_s
        delays = np.arange(low, high, self.coarse_step_s)

        # The fit only uses every n-th subcarrier, which is much cheaper and enough
        # to find the strongest path.
        every = self.fit_every
        basis = np.exp(-2j * np.pi * np.outer(frequencies[::every], delays))
        amplitudes = fit_sparse_paths(
            vectors[:, ::every], basis, self.lasso_weight, self.iterations
        )
        return as_real_array(delays[np.abs(amplitudes).argmax(axis=1)])

    def _rebuild_relative_to_strongest(
        self,
        vectors: ComplexArray,
        frequencies: RealArray,
        strongest_delays: RealArray,
    ) -> ComplexArray:
        """Refit on a fine grid, and rebuild each vector relative to its strongest."""

        rebuilt = np.zeros_like(vectors)

        # Vectors with the same coarse delay share one fine grid, so each such
        # group is refitted together.
        for coarse_delay in np.unique(strongest_delays):
            in_group = strongest_delays == coarse_delay
            rebuilt[in_group] = self._rebuild_group(
                vectors[in_group], frequencies, float(coarse_delay)
            )
        return rebuilt

    def _rebuild_group(
        self, vectors: ComplexArray, frequencies: RealArray, coarse_delay: float
    ) -> ComplexArray:
        """Rebuild vectors whose strongest path is near `coarse_delay`."""

        # The fine grid reaches from a bit before to a bit after the coarse delay,
        # and stays inside the allowed delay range.
        low, high = self.delay_range_s
        delays = np.arange(
            max(coarse_delay - self.fine_before_s, low),
            min(coarse_delay + self.fine_after_s, high),
            self.fine_step_s,
        )

        # Fit on every n-th subcarrier, but rebuild on all of them.
        every = self.fit_every
        basis = np.exp(-2j * np.pi * np.outer(frequencies, delays))
        amplitudes = fit_sparse_paths(
            vectors[:, ::every], basis[::every], self.lasso_weight, self.iterations
        )

        # The strongest path as it appears across the subcarriers. Multiplying by
        # its conjugate removes the phase that all paths share.
        rows = np.arange(len(amplitudes))
        strongest = np.abs(amplitudes).argmax(axis=1)
        strongest_path = basis[:, strongest].T * amplitudes[rows, strongest, None]
        return (amplitudes @ basis.T) * np.conj(strongest_path)
