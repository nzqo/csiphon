"""Concrete NumPy array types used at numerical boundaries."""

import numpy as np
import numpy.typing as npt

type RealArray = npt.NDArray[np.float64]
type ComplexArray = npt.NDArray[np.complex128]
type SignalArray = RealArray | ComplexArray


def as_complex_array(values: npt.ArrayLike) -> ComplexArray:
    """Return values as the canonical complex array type."""

    return np.asarray(values, dtype=np.complex128)


def as_real_array(values: npt.ArrayLike) -> RealArray:
    """Return values as the canonical real array type."""

    return np.asarray(values, dtype=np.float64)


def as_signal_array(values: npt.ArrayLike) -> SignalArray:
    """Preserve real or complex values while normalizing the dtype."""

    if np.iscomplexobj(values):
        return as_complex_array(values)

    return as_real_array(values)
