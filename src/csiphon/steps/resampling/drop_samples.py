"""Drop samples to simulate bad timing: independent loss, or bursty (Gilbert-Elliott).

An augmentation / robustness step: it throws packets away so a downstream `Resample`
(or the rest of a pipeline) can be tested against realistic timing loss. Survivors
keep their own real timestamps; only the *set* of samples changes. It is seeded, so
the same seed drops the same samples: deterministic despite being random.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from csiphon.core.arrays import as_real_array, as_signal_array
from csiphon.core.axes import AxisName
from csiphon.core.errors import LayoutError
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.signal import Signal
from csiphon.pipeline.step import Step
from csiphon.spec import Category, LayoutEffect, StepSpec, Streaming

BoolArray = npt.NDArray[np.bool_]


class LossModel(ABC):  # pylint: disable=too-few-public-methods
    """A packet-loss process: which of `count` samples survive.

    A one-method strategy (like the resampler fills): pick a subclass and hand it to
    `DropSamples(model=...)`. It only decides the keep/drop pattern; it never touches
    timestamps.
    """

    @abstractmethod
    def keep_mask(
        self, count: int, loss_rate: float, rng: np.random.Generator
    ) -> BoolArray:
        """A boolean mask of length `count`: True where the sample survives."""


@dataclass(frozen=True, slots=True)
class Independent(LossModel):  # pylint: disable=too-few-public-methods
    """Each sample is dropped on its own coin flip, with probability `loss_rate`."""

    def keep_mask(
        self, count: int, loss_rate: float, rng: np.random.Generator
    ) -> BoolArray:
        """Keep a sample when its uniform draw clears the loss rate."""

        return np.asarray(rng.random(count) >= loss_rate, dtype=bool)


@dataclass(frozen=True, slots=True)
class Bursty(LossModel):  # pylint: disable=too-few-public-methods
    """Gilbert-Elliott loss: drops come in runs averaging `mean_burst` samples.

    A two-state Markov chain: a "good" state that keeps and a "bad" state that
    drops. The bad runs last `mean_burst` samples on average (so `1/mean_burst` is the
    chance of recovering each step), and the long-run dropped fraction is `loss_rate`;
    the good->bad chance is derived from those two so the knobs stay intuitive.
    """

    mean_burst: float = field(
        default=2.0, metadata={"doc": "mean length of a loss burst, in samples"}
    )

    def __post_init__(self) -> None:
        """A burst is at least one sample, so its mean length must be >= 1."""

        if self.mean_burst < 1.0:
            raise LayoutError(f"mean_burst must be >= 1, got {self.mean_burst}.")

    def keep_mask(
        self, count: int, loss_rate: float, rng: np.random.Generator
    ) -> BoolArray:
        """Walk the good/bad chain: keep in the good state, drop in the bad one."""

        # Turn the burst length into the bad -> good chance per step, so a bad
        # run lasts mean_burst samples on average.
        recover = 1.0 / self.mean_burst
        # Solve loss_rate = to_bad / (to_bad + recover) for the good -> bad chance.
        to_bad = 0.0 if loss_rate <= 0 else loss_rate * recover / (1.0 - loss_rate)

        keep = np.ones(count, dtype=bool)
        # Start bad as often as the long-run drop fraction, so the walk needs no
        # warm-up.
        bad = bool(rng.random() < loss_rate)
        for index in range(count):
            keep[index] = not bad
            if rng.random() < (recover if bad else to_bad):
                bad = not bad
        return np.asarray(keep, dtype=bool)


@dataclass(frozen=True, slots=True)
class DropSamples(Step):
    """Drop samples to simulate timing loss (independent or bursty), reproducibly.

    Drops a `loss_rate` fraction of the samples using `model`, keeping the survivors'
    own timestamps. With `keep_endpoints` (the default) the first and last sample are
    always kept, so the recording still spans its full time range, handy before a
    Resample. `seed` fixes the randomness, so a given seed always drops the same set.
    Batch-only: it is an offline augmentation, and keeping the last sample needs the
    whole recording.
    """

    loss_rate: float = field(
        default=0.0, metadata={"doc": "fraction of samples to drop, in [0, 1)"}
    )
    model: LossModel = field(
        default_factory=Independent,
        metadata={"doc": "loss process (Independent or Bursty)"},
    )
    keep_endpoints: bool = field(
        default=True, metadata={"doc": "always keep the first and last sample"}
    )
    seed: int = field(default=0, metadata={"doc": "seed for reproducible dropping"})

    spec: ClassVar[StepSpec] = StepSpec(
        name="drop-samples",
        summary="drop samples to simulate independent or bursty timing loss",
        category=Category.RESAMPLING,
        admissible_values=None,
        admissible_reprs=None,
        requires_axes=(AxisName.TIME,),
        layout_effect=LayoutEffect(note="a random subset of the samples dropped"),
        streaming=Streaming.UNAVAILABLE,
        streaming_note="offline augmentation; the last-sample keep needs the full run",
    )

    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Time stays the (dynamic) axis; structure and semantics are unchanged."""

        self.require_inputs(layout)
        if not 0.0 <= self.loss_rate < 1.0:
            raise LayoutError(f"loss_rate must be in [0, 1), got {self.loss_rate}.")
        return layout

    def process(
        self, signal: Signal, out_layout: Layout, profile: AcquisitionProfile
    ) -> Signal:
        """Drop the chosen samples, keeping the survivors' real timestamps."""

        time_index = signal.layout.dynamic_index
        if time_index is None or signal.n_samples == 0:
            return signal.with_values(signal.values, out_layout)

        rng = np.random.default_rng(self.seed)
        keep = self.model.keep_mask(signal.n_samples, self.loss_rate, rng)
        if self.keep_endpoints:
            keep[0] = True
            keep[-1] = True

        indices = np.nonzero(keep)[0]
        values = as_signal_array(np.take(signal.values, indices, axis=time_index))
        times = as_real_array(signal.times[indices])
        return signal.with_values(values, out_layout, times=times)
