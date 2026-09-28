"""Splitting complex CSI into real and imaginary parts and joining them again."""

import numpy as np
import pytest
from conftest import stream_in_chunks

from csiphon import AcquisitionProfile, ComplexFromParts, Pipeline, Signal, ValueKind
from csiphon.core.errors import CompileError
from csiphon.steps import ButterworthFilter, ImagPart, Magnitude, RealPart


def _split(real: Pipeline | None = None, imag: Pipeline | None = None) -> Pipeline:
    """Branch into a real-part line `re` and an imaginary-part line `im`."""

    return Pipeline().branch(
        re=Pipeline().then(RealPart()) if real is None else real,
        im=Pipeline().then(ImagPart()) if imag is None else imag,
    )


def test_real_part_matches_numpy(
    profile: AcquisitionProfile, raw_signal: Signal
) -> None:
    """RealPart returns the real part and relabels the values as real-valued."""

    out = Pipeline().then(RealPart()).compile(profile).pour(raw_signal).single()
    assert out.layout.values == ValueKind.REAL
    assert out.layout.axis_names == raw_signal.layout.axis_names
    assert np.array_equal(out.values, raw_signal.values.real)


def test_imag_part_matches_numpy(
    profile: AcquisitionProfile, raw_signal: Signal
) -> None:
    """ImagPart returns the imaginary part and relabels the values as real-valued."""

    out = Pipeline().then(ImagPart()).compile(profile).pour(raw_signal).single()
    assert out.layout.values == ValueKind.REAL
    assert out.layout.axis_names == raw_signal.layout.axis_names
    assert np.array_equal(out.values, raw_signal.values.imag)


@pytest.mark.parametrize("step", [RealPart(), ImagPart()])
def test_parts_reject_real_input(profile: AcquisitionProfile, step: RealPart) -> None:
    """Taking a part of already real values is rejected at compile time."""

    with pytest.raises(CompileError, match=step.name):
        Pipeline().then(Magnitude()).then(step).compile(profile)


def test_complex_from_parts_rebuilds_the_signal(
    profile: AcquisitionProfile, raw_signal: Signal
) -> None:
    """Splitting and joining again returns the original complex CSI."""

    siphon = _split().merge(["re", "im"], using=ComplexFromParts()).compile(profile)
    out = siphon.pour(raw_signal).single()
    assert out.layout.values == ValueKind.COMPLEX
    assert np.array_equal(out.values, raw_signal.values)


def test_complex_from_parts_takes_the_first_branch_as_real(
    profile: AcquisitionProfile, raw_signal: Signal
) -> None:
    """The merge order decides the parts: listing `im` first makes it the real part."""

    siphon = _split().merge(["im", "re"], using=ComplexFromParts()).compile(profile)
    out = siphon.pour(raw_signal).single()
    expected = raw_signal.values.imag + 1j * raw_signal.values.real
    assert np.array_equal(out.values, expected)


def test_complex_from_parts_rejects_complex_branches(
    profile: AcquisitionProfile,
) -> None:
    """Both branches must be real-valued parts."""

    pipeline = _split(real=Pipeline(), imag=Pipeline()).merge(
        ["re", "im"], using=ComplexFromParts()
    )
    with pytest.raises(CompileError, match="real-valued"):
        pipeline.compile(profile)


def test_complex_from_parts_rejects_three_branches(
    profile: AcquisitionProfile,
) -> None:
    """Exactly two branches (real part, imaginary part) are joined."""

    pipeline = (
        Pipeline()
        .branch(
            re=Pipeline().then(RealPart()),
            im=Pipeline().then(ImagPart()),
            extra=Pipeline().then(RealPart()),
        )
        .merge(using=ComplexFromParts())
    )
    with pytest.raises(CompileError, match="exactly two"):
        pipeline.compile(profile)


def test_split_filter_equals_filtering_complex_input(
    profile: AcquisitionProfile, raw_signal: Signal
) -> None:
    """Filtering each part and joining them equals filtering the complex input."""

    high_pass = ButterworthFilter(cutoff_hz=5.0)
    split = (
        _split(
            real=Pipeline().then(RealPart()).then(high_pass),
            imag=Pipeline().then(ImagPart()).then(high_pass),
        )
        .merge(["re", "im"], using=ComplexFromParts())
        .compile(profile)
    )
    direct = Pipeline().then(high_pass).compile(profile)
    assert np.allclose(
        split.pour(raw_signal).single().values, direct.pour(raw_signal).single().values
    )


@pytest.mark.parametrize("chunk", [37, 128])
def test_complex_from_parts_streams_like_batch(
    profile: AcquisitionProfile, raw_signal: Signal, chunk: int
) -> None:
    """Split and join stream to the same values as a whole-recording pour."""

    siphon = _split().merge(["re", "im"], using=ComplexFromParts()).compile(profile)
    streamed = stream_in_chunks(siphon, raw_signal, chunk)
    assert np.array_equal(streamed.values, siphon.pour(raw_signal).single().values)
