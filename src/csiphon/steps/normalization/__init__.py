"""Normalization steps: rescale relative to a frame or an axis's statistics."""

from csiphon.steps.normalization.gain_normalize import GainNormalize
from csiphon.steps.normalization.global_max_normalize import GlobalMaxNormalize
from csiphon.steps.normalization.per_frame_max_normalize import PerFrameMaxNormalize

__all__ = [
    "GainNormalize",
    "GlobalMaxNormalize",
    "PerFrameMaxNormalize",
]
