"""Spec-driven contract checks that run automatically over every step.

Each step declares a StepSpec: the value kinds it accepts, the axes it needs,
and whether it can stream. These tests take that declaration and check the step
actually behaves that way -- without a hand-written test per step. The input for
each step is *derived from its own spec*, so adding a new step automatically
subjects it to the same checks.

What is checked here is the part of the contract that needs no English parsing:
  * the step accepts an input built to match its spec (it compiles and runs),
  * it rejects a value kind its spec excludes,
  * it rejects an input missing an axis its spec requires,
  * it honours its streaming declaration (refuse, or stream == batch).

The *shape* of the output ("adds a frequency axis", "subcarrier -> delay") is
still described in prose, so those claims are pinned by the bespoke tests at the
bottom of this file and in the per-topic test modules.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import concrete_steps, stream_in_chunks

from csiphon import (
    AcquisitionProfile,
    Axis,
    AxisName,
    Layout,
    Pipeline,
    Representation,
    Signal,
    ValueKind,
    create_signal,
    describe,
)
from csiphon.core import CompileError, StreamingError
from csiphon.pipeline.step import Step
from csiphon.spec import Deferred, StepSpec, Streaming
from csiphon.steps import (
    AxisReference,
    CovarianceSpectrum,
    FoldAxes,
    Hold,
    Multitaper,
    PrincipalComponents,
    Resample,
    RobustPca,
    SelectAxis,
    SubsampleEvery,
    SynchrosqueezedPower,
    TimeDifference,
)
from csiphon.steps.calibration import Reference
from csiphon.steps.reduction.principal_components import PCABasis, fit_pca_basis
from csiphon.steps.temporal_features.time_difference import Mode

# A small, fixed rig so band edges and frequency coordinates are predictable.
PROFILE = AcquisitionProfile(
    n_rx_antennas=3, subcarrier_indices=tuple(range(8)), sampling_rate_hz=1000.0
)
SAMPLES = 1024
NYQUIST = PROFILE.sampling_rate_hz / 2.0


# --- deriving a spec-valid input for an arbitrary step ------------------------


def _payload_axis(name: AxisName) -> Axis:
    """A static axis of `name` with coordinates a step can actually use."""

    if name == AxisName.SUBCARRIER:
        return Axis.sized(AxisName.SUBCARRIER, 8)
    if name == AxisName.RX_ANTENNA:
        return Axis.sized(AxisName.RX_ANTENNA, 3)
    if name == AxisName.DELAY:
        return Axis.static(AxisName.DELAY, tuple(float(i) for i in range(16)))
    if name == AxisName.FREQUENCY:
        coords = tuple(np.linspace(-NYQUIST, NYQUIST, 32, endpoint=False).tolist())
        return Axis.static(AxisName.FREQUENCY, coords, unit="Hz")
    if name == AxisName.FEATURE:
        return Axis.sized(AxisName.FEATURE, 6)
    return Axis.sized(name, 4)


def _input_layout(spec: StepSpec, value: ValueKind) -> Layout:
    """The most generic input the spec accepts, at the given value kind."""

    repr_ = (
        spec.admissible_reprs[0]
        if spec.admissible_reprs
        else Representation.CHANNEL_FREQUENCY_RESPONSE
    )
    axes = [Axis.dynamic(AxisName.TIME)]
    for name in spec.requires_axes:
        if name != AxisName.TIME:
            axes.append(_payload_axis(name))
    # Every step needs something to operate on besides time; give it subcarriers.
    if len(axes) == 1:
        axes.append(_payload_axis(AxisName.SUBCARRIER))
    return Layout(tuple(axes), repr_, value)


def _signal(layout: Layout) -> Signal:
    """Random data on `layout` with the dtype its value kind requires."""

    shape = [SAMPLES] + [axis.size for axis in layout.axes[1:]]
    rng = np.random.default_rng(0)
    if layout.values == ValueKind.COMPLEX:
        data = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    else:
        data = np.abs(rng.standard_normal(shape)) + 0.1
    return create_signal(data, np.arange(SAMPLES) / PROFILE.sampling_rate_hz, layout)


# --- the step registry and the few steps that need help to run ----------------


CONCRETE_STEPS = concrete_steps()
STEP_IDS = [cls.spec.name for cls in CONCRETE_STEPS]

# A few steps are meaningless with their default arguments (they must be told
# which axes/indices to act on). `_instance` returns a runnable configured step,
# and `_layout` the input it expects; everything else derives from the spec.
_FOLD_LAYOUT = Layout(
    (
        Axis.dynamic(AxisName.TIME),
        Axis.sized(AxisName.RX_ANTENNA, 3),
        Axis.sized(AxisName.SUBCARRIER, 8),
    ),
    Representation.CHANNEL_FREQUENCY_RESPONSE,
    ValueKind.MAGNITUDE,
)


def _instance(cls: type[Step]) -> Step:
    """A runnable, configured instance of `cls`."""

    if cls is FoldAxes:
        return FoldAxes(axes=(AxisName.RX_ANTENNA, AxisName.SUBCARRIER))
    if cls is SelectAxis:
        return SelectAxis(axis=AxisName.SUBCARRIER, indices=(0, 1))
    if cls is RobustPca:
        return RobustPca(num_components=2)
    if cls is SubsampleEvery:  # `every` is required
        return SubsampleEvery(every=2)
    if cls is Resample:  # needs a rate and a fill
        return Resample(rate_hz=100.0, fill=Hold())
    return cls()


# Steps that collapse "the feature axis" -- their effect is stated as FEATURE→X,
# so the harness feeds them a feature axis (not the generic subcarrier payload).
_FEATURE_INPUT = {
    "covariance-spectrum",
    "local-pca-bias",
    "multitaper-power",
    "robust-pca",
}


def _layout(cls: type[Step], value: ValueKind) -> Layout:
    """The input layout `cls` expects, at the given value kind."""

    if cls is FoldAxes:  # needs two axes to fold, not just one payload axis
        return Layout(_FOLD_LAYOUT.axes, _FOLD_LAYOUT.representation, value)
    if cls.spec.name in _FEATURE_INPUT:
        return Layout(
            (Axis.dynamic(AxisName.TIME), Axis.sized(AxisName.FEATURE, 6)),
            Representation.FEATURE_VECTOR,
            value,
        )
    return _input_layout(cls.spec, value)


def _strongest_value(step: Step) -> ValueKind:
    """The most demanding value kind this instance accepts (complex if any)."""

    values = step.resolve_admissible_values()
    return values[0] if values else ValueKind.COMPLEX


def _accepted_value(step: Step) -> ValueKind:
    """A value kind this instance accepts, for the non-acceptance tests."""

    values = step.resolve_admissible_values()
    return values[0] if values else ValueKind.MAGNITUDE


# Membership is decided from each step's *resolved* contract (the configured
# instance), not the static spec -- that is the whole point of the mechanism.
_ANY_VALUE_STEPS = [
    c for c in CONCRETE_STEPS if _instance(c).resolve_admissible_values() is None
]
_RESTRICTED_STEPS = [
    c for c in CONCRETE_STEPS if _instance(c).resolve_admissible_values() is not None
]
_AXIS_STEPS = [
    c
    for c in CONCRETE_STEPS
    if set(_instance(c).resolve_required_axes()) - {AxisName.TIME}
]


# --- the automatic contract checks --------------------------------------------


@pytest.mark.parametrize("cls", CONCRETE_STEPS, ids=STEP_IDS)
def test_step_accepts_the_input_its_contract_describes(cls: type[Step]) -> None:
    """A step compiles and runs on an input built to match its resolved contract.

    The input is the widest it accepts (complex when it accepts "any"), so this
    also checks the "accepts any value kind" claim is truthful.
    """

    step = _instance(cls)
    layout = _layout(cls, _strongest_value(step))
    out = Pipeline().then(step).compile(PROFILE, inlet=layout).pour(_signal(layout))

    # Whatever the step does, time survives and the output is a real layout.
    result = out.single()
    assert isinstance(result.layout, Layout)
    assert AxisName.TIME in result.layout.axis_names


@pytest.mark.parametrize(
    "value", [ValueKind.COMPLEX, ValueKind.MAGNITUDE], ids=["complex", "real"]
)
@pytest.mark.parametrize(
    "cls", _ANY_VALUE_STEPS, ids=[c.spec.name for c in _ANY_VALUE_STEPS]
)
def test_any_value_step_accepts_both_complex_and_real(
    cls: type[Step], value: ValueKind
) -> None:
    """A step whose resolved contract is "any value kind" must accept both forms.

    "Any" has to mean any: if a step only really handles complex, or only real,
    its contract is lying and this fails.
    """

    step = _instance(cls)
    layout = _layout(cls, value)
    Pipeline().then(step).compile(PROFILE, inlet=layout).pour(_signal(layout))


@pytest.mark.parametrize(
    "cls", _RESTRICTED_STEPS, ids=[c.spec.name for c in _RESTRICTED_STEPS]
)
def test_step_rejects_a_value_kind_outside_its_contract(cls: type[Step]) -> None:
    """A step with a restricted value set refuses a kind outside that set."""

    step = _instance(cls)
    allowed = set(step.resolve_admissible_values() or ())
    disallowed = next(kind for kind in ValueKind if kind not in allowed)

    with pytest.raises(CompileError, match=cls.spec.name):
        Pipeline().then(step).compile(PROFILE, inlet=_layout(cls, disallowed))


@pytest.mark.parametrize("cls", _AXIS_STEPS, ids=[c.spec.name for c in _AXIS_STEPS])
def test_step_requires_the_axes_its_contract_declares(cls: type[Step]) -> None:
    """Dropping a required (non-time) axis makes the step refuse at compile."""

    step = _instance(cls)
    full = _layout(cls, _accepted_value(step))
    required = next(a for a in step.resolve_required_axes() if a != AxisName.TIME)

    # Rebuild the input without the required axis; everything else stays valid.
    kept = tuple(axis for axis in full.axes if axis.name != required)
    if len(kept) == 1:  # keep a *different* payload axis, not the removed one
        filler = (
            AxisName.RX_ANTENNA
            if required != AxisName.RX_ANTENNA
            else AxisName.SUBCARRIER
        )
        kept = (*kept, _payload_axis(filler))
    stripped = Layout(kept, full.representation, full.values)

    with pytest.raises(CompileError, match=cls.spec.name):
        Pipeline().then(step).compile(PROFILE, inlet=stripped)


@pytest.mark.parametrize("cls", CONCRETE_STEPS, ids=STEP_IDS)
def test_step_honours_its_streaming_contract(cls: type[Step]) -> None:
    """UNAVAILABLE steps refuse to stream; BATCH_EQUIVALENT ones stream == batch.

    Streaming is read from the *resolved* contract, so a configuration that
    cannot stream (a fit-on-data reducer, an SST with no block size) is required
    to refuse, and one that can is required to match batch.
    """

    step = _instance(cls)
    layout = _layout(cls, _accepted_value(step))
    compiled = Pipeline().then(step).compile(PROFILE, inlet=layout)
    signal = _signal(layout)

    streaming = step.resolve_streaming()
    if streaming is Streaming.UNAVAILABLE:
        with pytest.raises(StreamingError, match=cls.spec.name):
            compiled.stream()
    elif streaming is Streaming.BATCH_EQUIVALENT:
        batch = compiled.pour(signal).single()
        streamed = stream_in_chunks(compiled, signal, chunk=256)
        assert np.allclose(batch.values, streamed.values)
        assert np.allclose(batch.times, streamed.times)
    else:  # BATCH_DIVERGENT: it may stream, just not identically to batch.
        compiled.stream()  # must not raise


@pytest.mark.parametrize("cls", CONCRETE_STEPS, ids=STEP_IDS)
def test_step_output_shape_matches_its_layout_effect(cls: type[Step]) -> None:
    """The real output layout matches the step's declared LayoutEffect.

    This is the auto-check the prose could never give: the axes a step says it
    adds actually appear, the ones it removes are gone, renames happen, and the
    value kind lands where declared. Sizes are not checked (config/runtime).
    """

    step = _instance(cls)
    effect = step.resolve_layout_effect()
    assert effect is not None, f"{cls.spec.name} has no structured layout_effect"

    layout = _layout(cls, _accepted_value(step))
    out = Pipeline().then(step).compile(PROFILE, inlet=layout).pour(_signal(layout))
    in_names = set(layout.axis_names)
    out_names = set(out.single().layout.axis_names)

    for added in effect.adds:
        assert added in out_names and added not in in_names, f"should add {added}"
    for removed in effect.removes:
        assert removed in in_names and removed not in out_names, (
            f"should drop {removed}"
        )
    for old, new in effect.replaces:
        assert old in in_names and old not in out_names, f"should consume {old}"
        assert new in out_names and new not in in_names, f"should produce {new}"

    out_values = out.single().layout.values
    if effect.value_kind is not None:
        assert out_values == effect.value_kind
    else:
        assert out_values == layout.values  # unchanged when not declared


# --- the static declaration may never lie about the resolved contract ---------


@pytest.mark.parametrize("cls", CONCRETE_STEPS, ids=STEP_IDS)
def test_static_declaration_agrees_with_the_resolved_contract(cls: type[Step]) -> None:
    """A concrete spec field must match what the instance resolves; a
    CONFIG_DEPENDENT field must resolve to a concrete value.

    This is what makes the tiers honest: a step cannot declare a concrete value
    and then quietly resolve to something else (that is exactly the trap the old
    `None`-means-any TimeDifference fell into), and it cannot leave a field
    CONFIG_DEPENDENT without actually resolving it.
    """

    step = _instance(cls)
    spec = cls.spec

    resolved_values = step.resolve_admissible_values()
    assert not isinstance(resolved_values, Deferred)  # resolution is always concrete
    if not isinstance(spec.admissible_values, Deferred):
        assert resolved_values == spec.admissible_values

    resolved_streaming = step.resolve_streaming()
    assert not isinstance(resolved_streaming, Deferred)
    if not isinstance(spec.streaming, Deferred):
        assert resolved_streaming == spec.streaming

    resolved_effect = step.resolve_layout_effect()
    assert not isinstance(resolved_effect, Deferred)
    if not isinstance(spec.layout_effect, Deferred):
        assert resolved_effect == spec.layout_effect


# --- per-instance resolved contract: the same class, two configurations -------


def test_time_difference_value_contract_depends_on_mode() -> None:
    """Conjugate modes resolve to complex-only; the plain difference to any."""

    assert TimeDifference(mode=Mode.CONJUGATE).resolve_admissible_values() == (
        ValueKind.COMPLEX,
    )
    assert TimeDifference(mode=Mode.PHASE_ONLY).resolve_admissible_values() == (
        ValueKind.COMPLEX,
    )
    assert TimeDifference(mode=Mode.DIFFERENCE).resolve_admissible_values() is None

    # The difference mode therefore runs on a real input; conjugate rejects it.
    layout = _input_layout(TimeDifference.spec, ValueKind.MAGNITUDE)
    Pipeline().then(TimeDifference(mode=Mode.DIFFERENCE)).compile(
        PROFILE, inlet=layout
    ).pour(_signal(layout))
    with pytest.raises(CompileError, match="time-difference"):
        Pipeline().then(TimeDifference(mode=Mode.CONJUGATE)).compile(
            PROFILE, inlet=layout
        )


def test_principal_components_streaming_depends_on_basis() -> None:
    """Fit-on-data PCA is batch-only; a pre-fit basis streams exactly like batch."""

    assert PrincipalComponents().resolve_streaming() is Streaming.UNAVAILABLE

    layout = Layout(
        (Axis.dynamic(AxisName.TIME), Axis.sized(AxisName.SUBCARRIER, 8)),
        Representation.CHANNEL_FREQUENCY_RESPONSE,
        ValueKind.MAGNITUDE,
    )
    signal = _signal(layout)
    basis = fit_pca_basis(signal, AxisName.SUBCARRIER, 2)
    configured = PrincipalComponents(n_components=2, basis=basis)
    assert configured.resolve_streaming() is Streaming.BATCH_EQUIVALENT

    compiled = Pipeline().then(configured).compile(PROFILE, inlet=layout)
    batch = compiled.pour(signal).single()
    streamed = stream_in_chunks(compiled, signal, chunk=128)
    assert np.allclose(batch.values, streamed.values)


def test_synchrosqueezed_streaming_depends_on_window() -> None:
    """SST is batch-only until a streaming_window turns on its block-local variant."""

    # resolve_streaming() is pure config -- it never touches the ssqueezepy backend.
    assert SynchrosqueezedPower().resolve_streaming() is Streaming.UNAVAILABLE
    windowed = SynchrosqueezedPower(streaming_window=256)
    assert windowed.resolve_streaming() is Streaming.BATCH_DIVERGENT

    layout = _input_layout(SynchrosqueezedPower.spec, ValueKind.REAL)
    Pipeline().then(windowed).compile(PROFILE, inlet=layout).stream()  # must not raise


def test_describe_reflects_the_resolved_contract_of_a_configured_step() -> None:
    """describe() shows the configured instance's contract, not the class default."""

    # The class itself cannot know, so it describes as config-dependent.
    assert describe(PrincipalComponents).as_dict()["streaming"] == "config-dependent"

    # A fit-on-data instance describes as batch-only.
    assert describe(PrincipalComponents()).as_dict()["streaming"] == "unavailable"

    # The same class with a basis describes as streamable.
    basis = PCABasis(mean=np.zeros(8), components=np.eye(8)[:2])
    described = describe(PrincipalComponents(n_components=2, basis=basis))
    assert described.as_dict()["streaming"] == "batch-equivalent"


# --- single-feature steps: operate on a chosen axis, broadcast the rest -------


def test_single_feature_step_broadcasts_over_an_extra_axis() -> None:
    """With more than the operated axis present, extra axes are parallel channels.

    A (time, rx_antenna, feature) input keeps rx_antenna and turns feature into the
    component axis -- the extra axis is broadcast over, not rejected.
    """

    three_axes = Layout(
        (
            Axis.dynamic(AxisName.TIME),
            Axis.sized(AxisName.RX_ANTENNA, 3),
            Axis.sized(AxisName.FEATURE, 6),
        ),
        Representation.FEATURE_VECTOR,
        ValueKind.REAL,
    )
    out = (
        Pipeline()
        .then(RobustPca(num_components=2))
        .compile(PROFILE, inlet=three_axes)
        .pour(_signal(three_axes))
        .single()
    )
    assert out.layout.axis_names == (
        AxisName.TIME,
        AxisName.RX_ANTENNA,
        AxisName.COMPONENT,
    )
    assert out.layout.axis(AxisName.RX_ANTENNA).size == 3
    assert out.values.shape == (SAMPLES, 3, 2)


def test_single_feature_step_operates_on_the_configured_axis() -> None:
    """The operated axis is config: name it to run on a (time, subcarrier) input."""

    subcarriers = Layout(
        (Axis.dynamic(AxisName.TIME), Axis.sized(AxisName.SUBCARRIER, 6)),
        Representation.FEATURE_VECTOR,
        ValueKind.REAL,
    )
    # The default axis is FEATURE, so a subcarrier input needs axis=SUBCARRIER.
    with pytest.raises(CompileError, match="feature"):
        Pipeline().then(RobustPca(num_components=2)).compile(PROFILE, inlet=subcarriers)
    out = (
        Pipeline()
        .then(RobustPca(num_components=2, axis=AxisName.SUBCARRIER))
        .compile(PROFILE, inlet=subcarriers)
        .pour(_signal(subcarriers))
        .single()
    )
    assert out.layout.axis(AxisName.COMPONENT).size == 2


def test_single_feature_effect_names_the_configured_axis() -> None:
    """CONFIG_DEPENDENT: the declared effect names the very axis the step consumes.

    This is the honesty the FEATURE-role placeholder lacked: give the step a
    subcarrier axis and its layout effect says subcarrier -> component/frequency,
    not a generic feature role.
    """

    assert RobustPca(axis=AxisName.SUBCARRIER).resolve_layout_effect().replaces == (
        (AxisName.SUBCARRIER, AxisName.COMPONENT),
    )
    assert Multitaper(axis=AxisName.SUBCARRIER).resolve_layout_effect().replaces == (
        (AxisName.SUBCARRIER, AxisName.FREQUENCY),
    )
    # Covariance renames a non-feature axis to its FEATURE stats axis, but reports a
    # plain resize when it already operates on the feature axis.
    subcarrier_effect = CovarianceSpectrum(
        axis=AxisName.SUBCARRIER
    ).resolve_layout_effect()
    assert subcarrier_effect.replaces == ((AxisName.SUBCARRIER, AxisName.FEATURE),)
    assert CovarianceSpectrum(
        axis=AxisName.FEATURE
    ).resolve_layout_effect().resizes == (AxisName.FEATURE,)


# --- CONFIG_DEPENDENT layout effects actually track the config ---------------


def test_select_axis_resize_effect_tracks_its_indices() -> None:
    """SelectAxis declares and performs a resize of the chosen axis to len(indices).

    Two different index counts must give two different declared/actual sizes, so the
    CONFIG_DEPENDENT effect genuinely follows the configuration.
    """

    layout = Layout(
        (Axis.dynamic(AxisName.TIME), Axis.sized(AxisName.SUBCARRIER, 8)),
        Representation.CHANNEL_FREQUENCY_RESPONSE,
        ValueKind.COMPLEX,
    )
    for indices in [(0, 1), (0, 1, 2, 3)]:
        step = SelectAxis(axis=AxisName.SUBCARRIER, indices=indices)
        assert step.resolve_layout_effect().resizes == (AxisName.SUBCARRIER,)
        out = step.output_layout(layout, PROFILE)
        assert out.axis(AxisName.SUBCARRIER).size == len(indices)


def test_axis_reference_resize_effect_tracks_its_mode() -> None:
    """AxisReference declares a resize exactly when its mode/drop_reference drops one.

    Plain FIXED keeps the axis; FIXED+drop_reference and ADJACENT each drop a position.
    The declared `resizes` must match what output_layout actually does in each case.
    """

    layout = Layout(
        (Axis.dynamic(AxisName.TIME), Axis.sized(AxisName.RX_ANTENNA, 3)),
        Representation.CHANNEL_FREQUENCY_RESPONSE,
        ValueKind.COMPLEX,
    )
    plain = AxisReference(axis=AxisName.RX_ANTENNA, mode=Reference.FIXED)
    dropping = AxisReference(
        axis=AxisName.RX_ANTENNA, mode=Reference.FIXED, drop_reference=True
    )
    adjacent = AxisReference(axis=AxisName.RX_ANTENNA, mode=Reference.ADJACENT)

    assert not plain.resolve_layout_effect().resizes  # keeps the axis
    assert dropping.resolve_layout_effect().resizes == (AxisName.RX_ANTENNA,)
    assert adjacent.resolve_layout_effect().resizes == (AxisName.RX_ANTENNA,)
    assert plain.output_layout(layout, PROFILE).axis(AxisName.RX_ANTENNA).size == 3
    assert dropping.output_layout(layout, PROFILE).axis(AxisName.RX_ANTENNA).size == 2
    assert adjacent.output_layout(layout, PROFILE).axis(AxisName.RX_ANTENNA).size == 2
