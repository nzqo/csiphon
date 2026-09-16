"""The step contract, the one extension point of the library.

A Step is one processing operation: an immutable config object (usually a frozen
dataclass) that does three things.

- output_layout(): compile time. Validate the incoming structure and describe
  the outgoing one. Touches no data.
- process(): batch. Run the full whole-recording algorithm. This is always the
  best-quality version; it never falls back to a streaming approximation.
- stream(): streaming. Hand back a stateful StreamOperator that eats chunks, or
  None when the step genuinely can't stream (a whole-recording mean subtraction,
  for example).

If your step transforms each time sample on its own, subclass PointwiseStep and
you get an exact streaming operator for free :)
"""

from abc import ABC, abstractmethod
from typing import ClassVar

from csiphon.core.arrays import SignalArray
from csiphon.core.axes import AxisName
from csiphon.core.layout import Layout
from csiphon.core.profile import AcquisitionProfile
from csiphon.core.semantics import ValueKind
from csiphon.core.signal import Signal, empty_signal
from csiphon.spec import Deferred, LayoutEffect, StepSpec, Streaming


class StreamOperator(ABC):
    """A stateful worker that processes a stream one chunk at a time."""

    @abstractmethod
    def push(self, chunk: Signal) -> Signal:
        """Consume one chunk and emit whatever output is ready (maybe empty)."""

    @abstractmethod
    def flush(self) -> Signal:
        """Emit any final buffered output at the end of the stream."""


class Step(ABC):
    """One validated structural transition plus its numerical operation.

    Every concrete step declares a StepSpec in the class attribute `spec`, its
    inspectable contract. `require_inputs` enforces exactly what the spec
    declares, so the documented requirements and the runtime checks never drift
    apart. Call `require_inputs` first thing in `output_layout` so the guards sit
    up front and the step fails early.
    """

    spec: ClassVar[StepSpec]

    @property
    def name(self) -> str:
        """Return this step's short identifier (from its spec)."""

        return self.spec.name

    def resolve_required_axes(self) -> tuple[AxisName, ...]:
        """Return the axes this configured step needs on its input.

        Steps whose required axis is chosen at construction (an `axis=`
        parameter) override this, since the class-level spec cannot know it.
        """

        return self.spec.requires_axes

    def resolve_admissible_values(self) -> tuple[ValueKind, ...] | None:
        """Return the value kinds this configured step accepts (None = any).

        Like `resolve_required_axes`, this refines the class-level spec for steps
        whose accepted value kinds depend on how they are configured (for example
        a step whose `mode` decides whether it needs complex input). A step whose
        spec marks this CONFIG_DEPENDENT must override this to give a concrete
        answer; otherwise the static spec value is returned.
        """

        values = self.spec.admissible_values
        if isinstance(values, Deferred):
            raise NotImplementedError(
                f"{self.spec.name}: admissible_values is CONFIG_DEPENDENT but "
                "resolve_admissible_values() is not overridden"
            )
        return values

    def resolve_streaming(self) -> Streaming:
        """Return how this configured step streams.

        Some steps can stream in one configuration and not another (a component
        step with a pre-fitted basis versus one that fits on the whole recording).
        A step whose spec marks streaming CONFIG_DEPENDENT must override this to
        give a concrete mode; otherwise the static spec value is returned.
        """

        streaming = self.spec.streaming
        if isinstance(streaming, Deferred):
            raise NotImplementedError(
                f"{self.spec.name}: streaming is CONFIG_DEPENDENT but "
                "resolve_streaming() is not overridden"
            )
        return streaming

    def resolve_layout_effect(self) -> LayoutEffect:
        """Return the structured layout change this configured step makes.

        Like the other resolve_* methods, a step whose spec marks the effect
        CONFIG_DEPENDENT (its axis change depends on constructor arguments) must
        override this to give a concrete effect.
        """

        effect = self.spec.layout_effect
        if isinstance(effect, Deferred):
            raise NotImplementedError(
                f"{self.spec.name}: layout_effect is CONFIG_DEPENDENT but "
                "resolve_layout_effect() is not overridden"
            )
        return effect

    def require_inputs(self, layout: Layout) -> None:
        """Enforce the spec's value-kind, representation, and axis rules."""

        admissible_values = self.resolve_admissible_values()
        if admissible_values is not None:
            layout.require_values(admissible_values)
        if self.spec.admissible_reprs is not None:
            layout.require_representation(self.spec.admissible_reprs)
        for axis in self.resolve_required_axes():
            layout.require_axis(axis)

    @abstractmethod
    def output_layout(self, layout: Layout, profile: AcquisitionProfile) -> Layout:
        """Validate the input layout and return the output layout."""

    @abstractmethod
    def process(
        self,
        signal: Signal,
        out_layout: Layout,
        profile: AcquisitionProfile,
    ) -> Signal:
        """Run the full batch algorithm on a complete signal."""

    def stream(  # pylint: disable=unused-argument  # overridable hook; subclasses use these
        self,
        in_layout: Layout,
        out_layout: Layout,
        profile: AcquisitionProfile,
    ) -> StreamOperator | None:
        """Return a streaming operator, or `None` if batch-only (default)."""

        return None


class PointwiseStep(Step):
    """A step that transforms each time sample independently.

    Because the operation does not mix information across time, applying it to
    a chunk is identical to applying it to the whole recording and slicing.
    Subclasses implement output_layout() and transform_values();
    both batch and (exact) streaming behaviour follow automatically.
    """

    @abstractmethod
    def transform_values(
        self,
        values: SignalArray,
        signal: Signal,
        out_layout: Layout,
        profile: AcquisitionProfile,
    ) -> SignalArray:
        """Transform a block of values without mixing across the time axis."""

    def process(
        self,
        signal: Signal,
        out_layout: Layout,
        profile: AcquisitionProfile,
    ) -> Signal:
        """Apply the pointwise transform to the whole signal."""

        new_values = self.transform_values(signal.values, signal, out_layout, profile)
        return signal.with_values(new_values, out_layout)

    def stream(
        self,
        in_layout: Layout,
        out_layout: Layout,
        profile: AcquisitionProfile,
    ) -> StreamOperator | None:
        """Return an exact per-chunk operator (never buffers)."""

        return _PointwiseOperator(self, out_layout, profile)


class _PointwiseOperator(StreamOperator):
    """Applies a PointwiseStep to each chunk with no state."""

    def __init__(
        self, step: PointwiseStep, out_layout: Layout, profile: AcquisitionProfile
    ) -> None:
        """Bind the step and the compiled output layout."""

        self._step = step
        self._out_layout = out_layout
        self._profile = profile

    def push(self, chunk: Signal) -> Signal:
        """Transform the chunk directly."""

        if chunk.n_samples == 0:
            return empty_signal(self._out_layout)
        return self._step.process(chunk, self._out_layout, self._profile)

    def flush(self) -> Signal:
        """Nothing is buffered, so emit an empty signal."""

        return empty_signal(self._out_layout)
