"""PCA dimensionality reduction over a chosen axis.

Produces a genuinely new kind of dimension: the outputs are abstract components,
not the original physical axis, so the axis is renamed to `COMPONENT` with
integer coordinates. Streamability depends on the basis: with a pre-fit basis the
projection is a per-sample linear map and streams exactly; fit-on-the-recording
PCA needs every sample, so it is batch-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import SignalArray
from csiphon.core.axes import Axis, AxisName
from csiphon.core.errors import LayoutError
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.signal import Signal, empty_signal
from csiphon.pipeline.step import Step, StreamOperator
from csiphon.spec import (
    CONFIG_DEPENDENT,
    Category,
    LayoutEffect,
    StepSpec,
    Streaming,
)


@dataclass(frozen=True, slots=True)
class PCABasis:
    """A pre-fit PCA basis: the training mean and the top component vectors."""

    # Real for real input; complex when fit on complex CSI (the projection in
    # `_project` stays consistent either way).
    # fmt: off
    mean       : SignalArray  # shape (n_features,)
    components : SignalArray  # shape (n_components, n_features)
    # fmt: on


def fit_pca_basis(signal: Signal, axis: AxisName, n_components: int) -> PCABasis:
    """Fit a PCA basis over `axis` using every sample in `signal`."""

    position = signal.layout.axis_position(axis)
    moved = np.moveaxis(signal.values, position, -1)
    # NOTE: Observation matrix shape is (observations, features).
    observations = moved.reshape(-1, moved.shape[-1])
    mean = observations.mean(axis=0)
    _, _, right_singular_vectors = np.linalg.svd(
        observations - mean, full_matrices=False
    )
    return PCABasis(mean=mean, components=right_singular_vectors[:n_components])


@dataclass(frozen=True, slots=True)
class PrincipalComponents(Step):
    """Reduce one axis to `n_components` PCA components.

    The output axis is AxisName.COMPONENT (abstract components, not the
    original physical dimension). With a pre-fit `basis` the step is a
    per-sample linear projection and streams exactly; without one it fits the
    basis on the whole recording, which is non-causal, so it is batch-only.
    """

    n_components: int = field(
        default=1, metadata={"doc": "number of components to keep"}
    )
    axis: AxisName = field(
        default=AxisName.SUBCARRIER, metadata={"doc": "axis to reduce"}
    )
    basis: PCABasis | None = field(
        default=None,
        compare=False,
        metadata={"doc": "pre-fit basis; if None, fit on the recording (batch-only)"},
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="principal-components",
        summary="reduce one axis to its top PCA components",
        category=Category.REDUCTION,
        admissible_values=None,
        admissible_reprs=None,
        # The required axis is chosen at run time; see resolve_required_axes.
        requires_axes=(),
        # Streams with a pre-fit basis, batch-only when fitting; see resolve_streaming.
        layout_effect=CONFIG_DEPENDENT,
        streaming=CONFIG_DEPENDENT,
        streaming_note="requires a pre-fit basis; fit-on-data is batch-only",
    )

    def resolve_required_axes(self) -> tuple[AxisName, ...]:
        """Require the configured axis to be present."""

        return (self.axis,)

    def resolve_streaming(self) -> Streaming:
        """Streams with a pre-fit basis; fitting on the recording is batch-only."""

        return Streaming.BATCH_EQUIVALENT if self.basis else Streaming.UNAVAILABLE

    def resolve_layout_effect(self) -> LayoutEffect:
        """Replace the configured axis with a component axis."""

        return LayoutEffect(
            replaces=((self.axis, AxisName.COMPONENT),),
            note="integer component coordinates",
        )

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Replace the source axis with a component axis of integer coordinates."""

        self.require_inputs(layout)
        if self.n_components < 1:
            raise LayoutError(f"n_components must be >= 1, got {self.n_components}.")
        source = layout.require_static_axis(self.axis)
        if source.size is not None and self.n_components > source.size:
            raise LayoutError(
                f"n_components={self.n_components} exceeds axis size {source.size}."
            )
        component = Axis.static(
            AxisName.COMPONENT, tuple(range(self.n_components)), unit="component"
        )
        return layout.replace_axis(self.axis, component)

    def process(
        self, signal: Signal, out_layout: Layout, profile: AcquisitionProfile
    ) -> Signal:
        """Project onto the components (fitting the basis first if needed)."""

        basis = self.basis
        if basis is None:
            basis = fit_pca_basis(signal, self.axis, self.n_components)
        position = signal.layout.axis_position(self.axis)
        projected = _project(signal.values, position, basis)
        return signal.with_values(projected, out_layout, coords=signal.coords)

    def stream(
        self, in_layout: Layout, out_layout: Layout, profile: AcquisitionProfile
    ) -> StreamOperator | None:
        """Stream only with a pre-fit basis; fit-on-data PCA is batch-only."""

        if self.basis is None:
            return None
        position = in_layout.axis_position(self.axis)
        return _ProjectOperator(self.basis, position, out_layout)


def _project(values: SignalArray, position: int, basis: PCABasis) -> SignalArray:
    """Center along `position` and project onto the basis components."""

    moved = np.moveaxis(values, position, -1)
    projected = (moved - basis.mean) @ basis.components.T
    return np.moveaxis(projected, -1, position)


class _ProjectOperator(StreamOperator):
    """Applies a fixed PCA projection to each chunk (exact, stateless)."""

    def __init__(self, basis: PCABasis, position: int, out_layout: Layout) -> None:
        """Bind the basis and the axis position to project."""

        self._basis = basis
        self._position = position
        self._out_layout = out_layout

    def push(self, chunk: Signal) -> Signal:
        """Project this chunk directly."""

        if chunk.n_samples == 0:
            return empty_signal(self._out_layout)
        projected = _project(chunk.values, self._position, self._basis)
        return chunk.with_values(projected, self._out_layout, coords=chunk.coords)

    def flush(self) -> Signal:
        """Nothing is buffered."""

        return empty_signal(self._out_layout)
