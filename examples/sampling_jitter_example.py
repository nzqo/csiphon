"""Non-uniform timestamps: csiphon tolerates jitter, warns when it is large.

WiFi CSI timestamps are never perfectly uniform. Steps use the profile's nominal
rate (or estimate an effective rate from the timestamps) and, when the jitter is
high, warn instead of silently trusting a uniform grid. Pass `strict=True` to turn
that warning into an error.

Runs with the base (numpy-only) install.
"""

# Examples share small profile/signal setup blocks by design.
# pylint: disable=duplicate-code
import warnings

import numpy as np

from csiphon import AcquisitionProfile, Pipeline
from csiphon.core import DataError, JitterWarning
from csiphon.core.sampling import effective_rate_hz
from csiphon.core.signal import Signal
from csiphon.steps import FixedSizeWindowSum, GainNormalize, Magnitude


def jittered_signal(profile: AcquisitionProfile, jitter: float) -> Signal:
    """A raw signal whose inter-sample intervals wobble by roughly `jitter`."""

    rng = np.random.default_rng(3)
    n = 800
    shape = (n, profile.n_rx_antennas, profile.n_subcarriers)
    csi = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    step = 1.0 / 1000.0
    times = np.cumsum(np.abs(step + rng.normal(0.0, jitter * step, n)))
    return profile.raw_signal(csi, times)


def main() -> None:
    """Run a jittery recording leniently (warns), then strictly (raises)."""

    profile = AcquisitionProfile(
        n_rx_antennas=3,
        subcarrier_indices=tuple(range(52)),
        sampling_rate_hz=1000.0,
    )
    signal = jittered_signal(profile, jitter=0.6)
    estimated = effective_rate_hz(signal.times, nominal_hz=None)
    print(f"nominal rate 1000.0 Hz; estimated from timestamps: {estimated:.1f} Hz")

    lenient = FixedSizeWindowSum(window_s=0.128, hop_s=0.05)
    compiled = (
        Pipeline()
        .then(Magnitude())
        .then(GainNormalize())
        .then(lenient)
        .compile(profile)
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled.pour(signal).single()
    warned = any(issubclass(w.category, JitterWarning) for w in caught)
    print("lenient run: warned about jitter ->", warned)

    strict = FixedSizeWindowSum(window_s=0.128, hop_s=0.05, strict=True)
    strict_pipeline = Pipeline().then(Magnitude()).then(GainNormalize()).then(strict)
    try:
        strict_pipeline.compile(profile).pour(signal).single()
    except DataError as error:
        print("strict run: DataError:", str(error).split(".", maxsplit=1)[0])


if __name__ == "__main__":
    main()
