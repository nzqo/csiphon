"""Scaling steps: a fixed pointwise scale or compression (log, decibels)."""

from __future__ import annotations

from csiphon.steps.scaling.log_scale import LogScale
from csiphon.steps.scaling.to_decibels import ToDecibels

__all__ = ["LogScale", "ToDecibels"]
