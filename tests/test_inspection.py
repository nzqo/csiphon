"""Every step carries a complete, inspectable contract.

These tests enforce the uniform, self-documenting structure: each concrete step
declares a full StepSpec, documents every parameter, and renders through
describe(). This is what keeps every step file consistent and instructive.
"""

from __future__ import annotations

import dataclasses

import pytest
from conftest import concrete_steps

from csiphon import Pipeline, describe
from csiphon.core import AxisName, CompileError
from csiphon.spec import Deferred, Description, LayoutEffect, StepSpec, Streaming
from csiphon.steps import (
    AxisReference,
    DyadicFrequencyBands,
    Magnitude,
    Phase,
    Power,
)

CONCRETE_STEPS = concrete_steps()


def test_there_are_steps() -> None:
    """Sanity check: the registry actually found the steps."""

    assert len(CONCRETE_STEPS) >= 20


@pytest.mark.parametrize("step_class", CONCRETE_STEPS, ids=lambda c: c.__name__)
def test_step_declares_a_complete_spec(step_class: type) -> None:
    """Every concrete step has a full StepSpec and documents each parameter."""

    spec = getattr(step_class, "spec", None)
    assert isinstance(spec, StepSpec), f"{step_class.__name__} has no StepSpec"
    assert spec.name and spec.summary and spec.category
    # Streaming and the layout effect are each a concrete value, or CONFIG_DEPENDENT.
    assert isinstance(spec.streaming, (Streaming, Deferred))
    assert isinstance(spec.layout_effect, (LayoutEffect, Deferred))

    for field in dataclasses.fields(step_class):
        doc = field.metadata.get("doc", "")
        assert doc, f"{step_class.__name__}.{field.name} has no 'doc' metadata"


def test_step_names_are_unique() -> None:
    """No two steps share a name (names identify steps in errors and describe)."""

    names = [cls.spec.name for cls in CONCRETE_STEPS]
    assert len(names) == len(set(names)), "duplicate step names"


@pytest.mark.parametrize("step_class", CONCRETE_STEPS, ids=lambda c: c.__name__)
def test_describe_renders_for_class_and_instance(step_class: type) -> None:
    """describe() works on both the class and a default-constructed instance."""

    class_description = describe(step_class)
    assert isinstance(class_description, Description)
    assert step_class.spec.name in str(class_description)

    instance = step_class()
    instance_description = describe(instance)
    assert step_class.spec.name in str(instance_description)
    assert instance_description.as_dict()["name"] == step_class.spec.name


def test_require_inputs_rejects_wrong_value_kind(profile) -> None:
    """A spec's value-kind requirement is actually enforced at compile time."""

    # Phase requires complex input; feeding magnitude must fail at compile.
    with pytest.raises(CompileError, match="phase"):
        Pipeline().then(Power()).then(Phase()).compile(profile)


def test_require_inputs_rejects_missing_axis(profile) -> None:
    """A spec's required axis is enforced at compile time."""

    with pytest.raises(CompileError, match="dyadic-frequency-bands"):
        Pipeline().then(Magnitude()).then(DyadicFrequencyBands()).compile(profile)


def test_describe_reflects_configured_axis() -> None:
    """describe(instance) shows the configured axis, not just the default."""

    description = describe(AxisReference(axis=AxisName.COMPONENT))
    assert AxisName.COMPONENT in description.requires_axes
