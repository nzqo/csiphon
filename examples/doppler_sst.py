"""Synchrosqueezed Doppler power: the SST front-end, needing the [sst] extra.

SynchrosqueezedPower is the only step here that pulls an optional dependency
(ssqueezepy). Install it with:  pip install 'csiphon[sst]'. Without the extra the
step raises MissingDependencyError at run time; the rest of the library is
numpy-only.

Its per-channel transform broadcasts over every channel axis, so the raw
(receiver, tx, rx, subcarrier) structure flows in directly -- no folding needed.
"""

# Examples share small profile/signal setup blocks by design.
# pylint: disable=duplicate-code
import numpy as np

from csiphon import AcquisitionProfile, Pipeline
from csiphon.core import MissingDependencyError
from csiphon.steps import GainNormalize, Magnitude, SynchrosqueezedPower


def main() -> None:
    """Build the SST pipeline and run it (if the extra is installed)."""

    profile = AcquisitionProfile(
        n_rx_antennas=3,
        subcarrier_indices=tuple(range(52)),
        sampling_rate_hz=1000.0,
    )
    compiled = (
        Pipeline().then(Magnitude()).then(GainNormalize()).then(SynchrosqueezedPower())
    ).compile(profile)
    print(compiled.describe())

    rng = np.random.default_rng(0)
    shape = (1500, profile.n_rx_antennas, profile.n_subcarriers)
    csi = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    signal = profile.raw_signal(csi, np.arange(shape[0]) / 1000.0)

    try:
        out = compiled.pour(signal).single()
    except MissingDependencyError as error:
        print("\nSST needs the [sst] extra: pip install 'csiphon[sst]'")
        print("  ", error)
        return
    print("\nSST features:", out.values.shape, out.layout.describe_axes())


if __name__ == "__main__":
    main()
