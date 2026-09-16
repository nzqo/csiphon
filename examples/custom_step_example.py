"""Bring your own step: write a custom step and drop it into a pipeline.

A step is just a class you subclass, no registration needed. Subclass
`PointwiseStep` when your operation treats each time sample on its own: you write
`output_layout` (the structural contract) and `transform_values` (the maths), and
both batch execution and exact streaming come for free. Subclass `Step` instead
when you must mix across time, and supply your own streaming operator.
"""

# Examples share small profile/signal setup blocks by design.
# pylint: disable=duplicate-code
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from csiphon import (
    AcquisitionProfile,
    Layout,
    Pipeline,
    PointwiseStep,
    Signal,
    StepSpec,
    Streaming,
    ValueKind,
    describe,
)
from csiphon.core.arrays import SignalArray, as_signal_array
from csiphon.spec import Category, LayoutEffect
from csiphon.steps import Magnitude


@dataclass(frozen=True, slots=True)
class ClipAbove(PointwiseStep):
    """Clip values above a threshold, leaving the layout untouched.

    It touches each sample independently, so streaming is exact and automatic.
    """

    threshold: float = field(default=1.0, metadata={"doc": "largest value to keep"})

    spec: ClassVar[StepSpec] = StepSpec(
        # The meta informative stuff first:
        name="clip-above",
        summary="clip values above a threshold",
        category=Category.CLEANING,
        # The actual requirements or admissible inputs:
        admissible_values=(ValueKind.MAGNITUDE, ValueKind.REAL, ValueKind.POWER),
        admissible_reprs=None,
        requires_axes=(),
        layout_effect=LayoutEffect(note="values clipped to a threshold"),
        streaming=Streaming.BATCH_EQUIVALENT,
    )

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Guards first (enforce the contract), then keep the same layout."""

        self.require_inputs(layout)
        return layout

    def transform_values(
        self,
        values: SignalArray,
        signal: Signal,
        out_layout: Layout,
        profile: AcquisitionProfile,
    ) -> SignalArray:
        """Clamp every value to at most `threshold`."""

        return as_signal_array(np.minimum(values, self.threshold))


def main() -> None:
    """Use the custom step exactly like a built-in one."""

    print(describe(ClipAbove(threshold=2.0)))  # your step is inspectable, too

    profile = AcquisitionProfile(
        n_rx_antennas=3,
        subcarrier_indices=tuple(range(52)),
        sampling_rate_hz=1000.0,
    )
    pipeline = Pipeline().then(Magnitude()).then(ClipAbove(threshold=2.0))
    compiled = pipeline.compile(profile)

    rng = np.random.default_rng(0)
    shape = (2000, profile.n_rx_antennas, profile.n_subcarriers)
    csi = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    times = np.arange(shape[0]) / 1000.0
    signal = profile.raw_signal(csi, times)

    out = compiled.pour(signal).single()
    print("\nmax value after clipping:", float(out.values.max()), "(threshold 2.0)")


if __name__ == "__main__":
    main()
