"""Inspect a step or a whole pipeline.

`describe(DelayAutocorrelation)` (a class), `describe(some_step_instance)`, or
`describe(siphon)` / `describe(pipeline)` returns a Description /
PipelineDescription: a readable, colorized block, and (via `.as_dict()` /
`.save(...)`) the same content as data you can store as metadata. This lives in
its own module so it can import Step and Pipeline without a cycle (spec stays
step-agnostic).
"""

from __future__ import annotations

from typing import overload

from csiphon.core.semantics import ValueKind
from csiphon.pipeline.pipeline import Pipeline, Siphon
from csiphon.pipeline.step import Step
from csiphon.spec import (
    Deferred,
    Description,
    LayoutEffect,
    PipelineDescription,
    Streaming,
    params_of,
)


@overload
def describe(target: Pipeline | Siphon) -> PipelineDescription: ...
@overload
def describe(target: type[Step] | Step) -> Description: ...
def describe(
    target: type[Step] | Step | Pipeline | Siphon,
) -> Description | PipelineDescription:
    """Inspect a step (class or instance) or a whole pipeline."""

    if isinstance(target, Siphon | Pipeline):
        return target.to_description()

    instance = None if isinstance(target, type) else target
    spec = (target if isinstance(target, type) else type(target)).spec
    # A configured instance reports its resolved contract; a class falls back to
    # the static spec (there is no configuration to resolve against yet).
    admissible_values: tuple[ValueKind, ...] | Deferred | None
    streaming: Streaming | Deferred
    layout_effect: LayoutEffect | Deferred
    if instance is not None:
        requires_axes = tuple(instance.resolve_required_axes())
        admissible_values = instance.resolve_admissible_values()
        streaming = instance.resolve_streaming()
        layout_effect = instance.resolve_layout_effect()
    else:
        requires_axes = spec.requires_axes
        admissible_values = spec.admissible_values
        streaming = spec.streaming
        layout_effect = spec.layout_effect
    return Description(
        spec=spec,
        params=params_of(target),
        requires_axes=requires_axes,
        admissible_values=admissible_values,
        streaming=streaming,
        layout_effect=layout_effect,
    )
