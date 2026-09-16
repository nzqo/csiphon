"""Fuse: combine branches that differ on one axis into a single feature vector."""

import numpy as np
import pytest

from csiphon.core.axes import Axis, AxisName
from csiphon.core.errors import LayoutError
from csiphon.core.layout import Layout
from csiphon.core.semantics import Representation, ValueKind
from csiphon.pipeline.merges import Fuse


def _layout(axes, representation, values) -> Layout:
    """Build a layout from (name, size) axis specs."""

    built = tuple(
        Axis.dynamic(name) if size is None else Axis.sized(name, size)
        for name, size in axes
    )
    return Layout(built, representation, values)


def test_fuse_folds_the_differing_axis_into_features() -> None:
    """Two differently named axes become one feature axis of the summed size."""

    sub = _layout(
        [(AxisName.TIME, None), (AxisName.SUBCARRIER, 4)],
        Representation.CHANNEL_FREQUENCY_RESPONSE,
        ValueKind.MAGNITUDE,
    )
    delay = _layout(
        [(AxisName.TIME, None), (AxisName.DELAY, 3)],
        Representation.DELAY_RESPONSE,
        ValueKind.MAGNITUDE,
    )
    out = Fuse().output_layout([sub, delay])

    assert out.axis_names == (AxisName.TIME, AxisName.FEATURE)
    assert out.axis(AxisName.FEATURE).size == 7  # 4 + 3
    assert out.representation is Representation.FEATURE_VECTOR
    assert out.values is ValueKind.MAGNITUDE  # both branches agreed


def test_fuse_unifies_mixed_value_kinds_to_real() -> None:
    """Magnitude and power are stackable: the fused features are plain real."""

    mag = _layout([(AxisName.SUBCARRIER, 4)], Representation.RATIO, ValueKind.MAGNITUDE)
    power = _layout([(AxisName.DELAY, 2)], Representation.RATIO, ValueKind.POWER)
    out = Fuse().output_layout([mag, power])

    assert out.values is ValueKind.REAL


def test_fuse_lifts_real_branch_into_complex() -> None:
    """With a complex branch present the result is complex; real branches lift."""

    mag = _layout(
        [(AxisName.TIME, None), (AxisName.SUBCARRIER, 2)],
        Representation.CHANNEL_FREQUENCY_RESPONSE,
        ValueKind.MAGNITUDE,
    )
    cplx = _layout(
        [(AxisName.TIME, None), (AxisName.DELAY, 2)],
        Representation.DELAY_RESPONSE,
        ValueKind.COMPLEX,
    )
    assert Fuse().output_layout([mag, cplx]).values is ValueKind.COMPLEX

    a = np.ones((3, 2), dtype=float)  # magnitude branch
    b = 1j * np.ones((3, 2), dtype=complex)  # complex branch
    fused = Fuse().combine([a, b], [mag, cplx])
    assert fused.dtype == np.complex128
    assert fused.shape == (3, 4)
    # The real branch sits in the real part; the complex branch keeps its values.
    assert np.allclose(fused[:, :2], 1 + 0j)
    assert np.allclose(fused[:, 2:], 1j)


def test_fuse_matches_axes_by_name_not_position() -> None:
    """Shared axes are lined up by name, so a reordered branch is transposed to fit."""

    first = _layout(
        [(AxisName.TIME, None), (AxisName.RX_ANTENNA, 2), (AxisName.SUBCARRIER, 4)],
        Representation.CHANNEL_FREQUENCY_RESPONSE,
        ValueKind.MAGNITUDE,
    )
    # Same axes by name, but rx and the fused axis are in the other order.
    reordered = _layout(
        [(AxisName.TIME, None), (AxisName.DELAY, 3), (AxisName.RX_ANTENNA, 2)],
        Representation.DELAY_RESPONSE,
        ValueKind.MAGNITUDE,
    )
    out = Fuse().output_layout([first, reordered])
    assert out.axis_names == (AxisName.TIME, AxisName.RX_ANTENNA, AxisName.FEATURE)

    a = np.arange(5 * 2 * 4).reshape(5, 2, 4).astype(float)
    b = np.arange(5 * 3 * 2).reshape(5, 3, 2).astype(float)  # (time, delay, rx)
    fused = Fuse().combine([a, b], [first, reordered])

    # b is transposed (time, delay, rx) -> (time, rx, delay) before concatenation.
    expected = np.concatenate([a, np.transpose(b, (0, 2, 1))], axis=2)
    assert fused.shape == (5, 2, 7)
    assert np.allclose(fused, expected)


def test_fuse_rejects_more_than_one_differing_axis() -> None:
    """If branches differ on two axes there is no single axis to fuse."""

    one = _layout(
        [(AxisName.RX_ANTENNA, 2), (AxisName.SUBCARRIER, 4)],
        Representation.RATIO,
        ValueKind.MAGNITUDE,
    )
    two = _layout(
        [(AxisName.RECEIVER, 2), (AxisName.DELAY, 4)],
        Representation.RATIO,
        ValueKind.MAGNITUDE,
    )
    with pytest.raises(LayoutError, match="unshared"):
        Fuse().output_layout([one, two])


# --- edge cases ---


def test_fuse_rejects_identical_branches() -> None:
    """Fuse needs exactly one differing axis; identical branches have none to fuse."""

    same = _layout(
        [(AxisName.TIME, None), (AxisName.SUBCARRIER, 4)],
        Representation.RATIO,
        ValueKind.MAGNITUDE,
    )
    with pytest.raises(LayoutError, match="unshared"):
        Fuse().output_layout([same, same])


def test_fuse_combines_the_same_axis_at_different_sizes() -> None:
    """Branches sharing an axis name but not its size fuse along that axis."""

    small = _layout(
        [(AxisName.TIME, None), (AxisName.SUBCARRIER, 4)],
        Representation.RATIO,
        ValueKind.MAGNITUDE,
    )
    large = _layout(
        [(AxisName.TIME, None), (AxisName.SUBCARRIER, 6)],
        Representation.RATIO,
        ValueKind.MAGNITUDE,
    )
    out = Fuse().output_layout([small, large])
    assert out.axis_names == (AxisName.TIME, AxisName.FEATURE)
    assert out.axis(AxisName.FEATURE).size == 10  # 4 + 6


def test_fuse_feature_axis_is_dynamic_when_a_fused_axis_is() -> None:
    """If a branch's fused axis is runtime-sized, the feature axis is too."""

    static = _layout(
        [(AxisName.TIME, None), (AxisName.SUBCARRIER, 4)],
        Representation.RATIO,
        ValueKind.MAGNITUDE,
    )
    dynamic = _layout(
        [(AxisName.TIME, None), (AxisName.FREQUENCY, None)],
        Representation.TIME_FREQUENCY,
        ValueKind.POWER,
    )
    assert Fuse().output_layout([static, dynamic]).axis(AxisName.FEATURE).size is None


def test_fuse_rejects_a_feature_name_that_already_exists() -> None:
    """The new feature axis must not collide with an axis the branches already share."""

    a = _layout(
        [(AxisName.FEATURE, 2), (AxisName.SUBCARRIER, 4)],
        Representation.FEATURE_VECTOR,
        ValueKind.REAL,
    )
    b = _layout(
        [(AxisName.FEATURE, 2), (AxisName.DELAY, 3)],
        Representation.FEATURE_VECTOR,
        ValueKind.REAL,
    )
    with pytest.raises(LayoutError, match="already exists"):
        Fuse().output_layout([a, b])


def test_fuse_across_three_branches_concatenates_in_order() -> None:
    """Fuse lays every branch's differing axis end to end, in branch order."""

    a = _layout(
        [(AxisName.TIME, None), (AxisName.SUBCARRIER, 2)],
        Representation.RATIO,
        ValueKind.MAGNITUDE,
    )
    b = _layout(
        [(AxisName.TIME, None), (AxisName.DELAY, 3)],
        Representation.DELAY_RESPONSE,
        ValueKind.MAGNITUDE,
    )
    c = _layout(
        [(AxisName.TIME, None), (AxisName.FREQUENCY, 4)],
        Representation.TIME_FREQUENCY,
        ValueKind.POWER,
    )
    assert Fuse().output_layout([a, b, c]).axis(AxisName.FEATURE).size == 9  # 2+3+4

    arrays = [np.full((5, 2), 1.0), np.full((5, 3), 2.0), np.full((5, 4), 3.0)]
    fused = Fuse().combine(arrays, [a, b, c])
    assert fused.shape == (5, 9)
    assert np.allclose(fused[:, :2], 1.0)
    assert np.allclose(fused[:, 2:5], 2.0)
    assert np.allclose(fused[:, 5:], 3.0)
