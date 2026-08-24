"""Robust PCA (PCP): decompose into low-rank + sparse, then project."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar

import numpy as np

from csiphon.core.arrays import RealArray, as_real_array, as_signal_array
from csiphon.core.axes import Axis, AxisName
from csiphon.core.errors import LayoutError
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.semantics import ValueKind
from csiphon.core.signal import Signal
from csiphon.pipeline.step import Step
from csiphon.spec import CONFIG_DEPENDENT, Category, LayoutEffect, StepSpec, Streaming
from csiphon.steps._support.broadcasting import (
    broadcast_channels,
    operated_axis_position,
)


class Source(StrEnum):
    """Which part of the low-rank + sparse split to build the basis from."""

    # fmt: off
    LOW_RANK = "lowrank"
    SPARSE   = "sparse"
    # fmt: on


def _soft_threshold(matrix: RealArray, tau: float) -> RealArray:
    """Elementwise soft-thresholding (shrinkage toward zero)."""

    return as_real_array(np.sign(matrix) * np.maximum(np.abs(matrix) - tau, 0.0))


def _svd_threshold(matrix: RealArray, tau: float) -> RealArray:
    """Singular-value thresholding: shrink each singular value by `tau`."""

    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    return as_real_array((left * np.maximum(singular - tau, 0.0)) @ right)


def _pcp_ialm(  # pylint: disable=too-many-locals  # a standard RPCA solver
    data: RealArray, *, rho: float, tol: float, max_iter: int
) -> tuple[RealArray, RealArray]:
    """Robust PCA via Principal Component Pursuit, solved with inexact ALM."""

    rows, cols = data.shape
    frobenius = float(np.linalg.norm(data, ord="fro"))
    if frobenius == 0.0:
        return np.zeros_like(data), np.zeros_like(data)

    # Standard PCP setup: lam balances the low-rank and sparse parts; the rest
    # initializes the dual variable and the penalty weight the ALM loop uses.
    lam = 1.0 / np.sqrt(max(rows, cols))
    spectral = float(np.linalg.norm(data, ord=2))
    infinity = float(np.linalg.norm(data.ravel(), ord=np.inf)) / max(lam, 1e-12)
    dual = np.asarray(data / max(spectral, infinity, 1e-12))
    penalty = 1.25 / max(spectral, 1e-12)

    low_rank = np.zeros_like(data)
    sparse = np.zeros_like(data)
    for _ in range(int(max_iter)):
        # Update the low-rank part by shrinking its singular values and the sparse
        # part by soft-thresholding, then take a dual (gradient) step.
        low_rank = _svd_threshold(data - sparse + dual / penalty, 1.0 / penalty)
        sparse = _soft_threshold(data - low_rank + dual / penalty, lam / penalty)
        residual = data - low_rank - sparse
        dual = dual + penalty * residual

        # Stop once low_rank + sparse reconstructs the data closely enough.
        if float(np.linalg.norm(residual, ord="fro")) / frobenius < tol:
            break
        penalty = min(penalty * rho, 1e7)
    return as_real_array(low_rank), as_real_array(sparse)


@dataclass(frozen=True, slots=True)
class RobustPca(Step):  # pylint: disable=too-many-instance-attributes  # RPCA has many knobs
    """Robust PCA (PCP): split into low-rank + sparse, then project onto one.

    Decomposes the whole recording `D = L + S` (low-rank background + sparse
    events) by the inexact-ALM method, learns a basis from the top components of
    `L` (`use="lowrank"`) or `S` (`use="sparse"`), and projects the data onto it.
    Unlike PrincipalComponents, the decomposition separates structured motion
    from outliers. Fits on the whole recording, so it is batch-only. Replaces the
    chosen `axis` with `component[K]`, running once per any other axis present.
    """

    axis: AxisName = field(
        default=AxisName.FEATURE, metadata={"doc": "axis to decompose along"}
    )
    num_components: int = field(
        default=12, metadata={"doc": "number of components to keep"}
    )
    first_component: int = field(
        default=0, metadata={"doc": "first component index to keep"}
    )
    fit_max_samples: int | None = field(
        default=20_000,
        metadata={"doc": "subsample to at most this many rows when fitting"},
    )
    rho: float = field(default=1.5, metadata={"doc": "ALM penalty growth factor"})
    tol: float = field(
        default=1e-7, metadata={"doc": "relative-residual convergence tolerance"}
    )
    max_iter: int = field(default=1000, metadata={"doc": "maximum ALM iterations"})
    center: bool = field(
        default=True, metadata={"doc": "subtract the per-feature mean first"}
    )
    use: Source = field(
        default=Source.LOW_RANK,
        metadata={"doc": "build the basis from the low-rank or sparse part"},
    )

    spec: ClassVar[StepSpec] = StepSpec(
        name="robust-pca",
        summary="project onto robust-PCA (low-rank + sparse) components",
        category=Category.REDUCTION,
        admissible_values=(ValueKind.MAGNITUDE, ValueKind.REAL, ValueKind.POWER),
        admissible_reprs=None,
        # The axis to decompose along is chosen at run time; see resolve_* below.
        requires_axes=(),
        layout_effect=CONFIG_DEPENDENT,  # replaces the chosen axis; see resolve below
        streaming=Streaming.UNAVAILABLE,
        streaming_note="fits the decomposition on the whole recording, so batch-only",
    )

    def resolve_required_axes(self) -> tuple[AxisName, ...]:
        """Require a time axis and the configured axis to decompose along."""

        return (AxisName.TIME, self.axis)

    def resolve_layout_effect(self) -> LayoutEffect:
        """The chosen axis becomes an abstract component axis."""

        return LayoutEffect(
            replaces=((self.axis, AxisName.COMPONENT),), value_kind=ValueKind.REAL
        )

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Replace the chosen axis with `num_components` abstract components."""

        self.require_inputs(layout)
        feature = layout.require_static_axis(self.axis)
        if self.num_components <= 0:
            raise LayoutError("RobustPca needs num_components >= 1.")
        if self.first_component < 0:
            raise LayoutError(
                f"first_component must be >= 0, got {self.first_component}."
            )
        if self.fit_max_samples is not None and self.fit_max_samples < 1:
            raise LayoutError(
                f"fit_max_samples must be >= 1, got {self.fit_max_samples}."
            )
        if (
            feature.size is not None
            and self.first_component + self.num_components > feature.size
        ):
            raise LayoutError(
                "first_component + num_components exceeds the feature size."
            )
        component = Axis.static(
            AxisName.COMPONENT, tuple(range(self.num_components)), unit="component"
        )
        return layout.replace_axis(self.axis, component).with_values(ValueKind.REAL)

    def process(
        self, signal: Signal, out_layout: Layout, profile: AcquisitionProfile
    ) -> Signal:
        """Fit the decomposition, learn a basis, and project the whole recording."""

        time_index = signal.layout.dynamic_index
        if time_index is None:
            raise ValueError("RobustPca requires a time axis.")
        # Put time on the first axis, then project the (time, axis) matrix once per
        # any other axis, so extra axes (receivers, antennas, ...) broadcast.
        data = as_signal_array(np.moveaxis(signal.values, time_index, 0))
        feature_pos = operated_axis_position(signal.layout, self.axis)
        scores = broadcast_channels(
            data, feature_pos, lambda matrix: self._project(as_real_array(matrix))
        )
        restored = np.moveaxis(scores, 0, time_index)
        return signal.with_values(as_signal_array(restored), out_layout)

    def _project(self, data: RealArray) -> RealArray:
        """Decompose, learn a basis from L or S, and return the scores `(T, K)`."""

        rows = data.shape[0]
        if self.fit_max_samples is None:
            fit_data = data
        else:
            indices = np.linspace(
                0, rows - 1, min(rows, self.fit_max_samples), dtype=int
            )
            fit_data = data[indices]

        mean = (
            fit_data.mean(axis=0, keepdims=True)
            if self.center
            else np.zeros((1, data.shape[1]))
        )
        low_rank, sparse = _pcp_ialm(
            as_real_array(fit_data - mean),
            rho=self.rho,
            tol=self.tol,
            max_iter=self.max_iter,
        )
        # Take the top components of the chosen part as the basis, then project
        # the full centered data onto it.
        source = low_rank if self.use == Source.LOW_RANK else sparse
        _, _, right = np.linalg.svd(source, full_matrices=False)
        basis = right[
            self.first_component : self.first_component + self.num_components
        ].T
        return as_real_array((data - mean) @ basis)
