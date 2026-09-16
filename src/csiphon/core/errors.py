"""Exceptions raised by layouts, steps, and pipeline compilation."""


class LayoutError(ValueError):
    """A signal layout does not satisfy a step's structural requirements."""


class DataError(ValueError):
    """Numerical values violate an assumption checked during execution."""


class CompileError(ValueError):
    """A pipeline step is incompatible with the layout before it."""


class StreamingError(ValueError):
    """A pipeline (or one of its steps) cannot run in streaming mode."""


class ClogError(StreamingError):
    """A merge branch held too many samples without aligning (a stalled branch)."""


class MissingDependencyError(ImportError):
    """A step needs an optional dependency that is not installed."""

    def __init__(self, package: str, extra: str) -> None:
        """Explain which extra to install for the missing dependency."""

        super().__init__(
            f"This step requires the optional dependency '{package}'. "
            f"Install it with:  pip install 'csiphon[{extra}]'"
        )
