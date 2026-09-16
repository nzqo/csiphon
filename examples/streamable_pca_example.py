"""Streamable PCA: a pre-fit basis turns PrincipalComponents into a per-sample map.

Fit-on-the-recording PCA needs every sample, so it is batch-only. But if you fit
the basis once offline and pass it in, the projection is a fixed linear map that
streams exactly, chunk by chunk, matching the batch result.

Runs with the base (numpy-only) install.
"""

# Examples share small profile/signal setup blocks by design.
# pylint: disable=duplicate-code
import numpy as np

from csiphon import AcquisitionProfile, AxisName, Pipeline, Signal
from csiphon.pipeline import concat_signals
from csiphon.steps import Magnitude, PrincipalComponents, fit_pca_basis


def make_signal(profile: AcquisitionProfile, n: int, seed: int) -> Signal:
    """A synthetic raw-CSI recording of `n` samples."""

    rng = np.random.default_rng(seed)
    shape = (n, profile.n_rx_antennas, profile.n_subcarriers)
    csi = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return profile.raw_signal(csi, np.arange(n) / 1000.0)


def main() -> None:
    """Fit a basis offline, then show the projection streams == batch."""

    profile = AcquisitionProfile(
        n_rx_antennas=3,
        subcarrier_indices=tuple(range(52)),
        sampling_rate_hz=1000.0,
    )

    # Fit the basis once, offline, over the subcarrier axis of training data.
    magnitude = Pipeline().then(Magnitude()).compile(profile)
    training = magnitude.pour(make_signal(profile, 2000, seed=0)).single()
    basis = fit_pca_basis(training, AxisName.SUBCARRIER, n_components=4)

    # With that basis, PrincipalComponents is a fixed map -> it streams.
    compiled = (
        Pipeline()
        .then(Magnitude())
        .then(
            PrincipalComponents(axis=AxisName.SUBCARRIER, n_components=4, basis=basis)
        )
    ).compile(profile)

    live = make_signal(profile, 500, seed=1)
    batch = compiled.pour(live).single()

    stream = compiled.stream()
    outputs = []
    for start in range(0, live.n_samples, 64):
        piece = Signal(
            values=live.values[start : start + 64],
            times=live.times[start : start + 64],
            layout=live.layout,
        )
        outputs.append(stream.flow(piece).single())
    outputs.append(stream.flush().single())
    streamed = concat_signals(outputs, compiled.outlet_layout)

    print("output layout:", batch.layout.describe_axes())
    print("streamed equals batch:", np.allclose(batch.values, streamed.values))


if __name__ == "__main__":
    main()
