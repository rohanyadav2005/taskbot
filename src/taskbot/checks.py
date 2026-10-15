"""Shared result type returned by every engineering check and simulation test."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, field_validator


class CheckResult(BaseModel):
    """Outcome of one check, in SI units.

    `message` is fed back to the LLM verbatim, so it must say what failed,
    by how much, and what to change (e.g. "shoulder_pitch needs 12.4 N·m but
    the selected motor gives 8.0 N·m; choose a stronger motor or shorten the
    arm").
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    check: str
    passed: bool
    value: float | None = None
    limit: float | None = None
    message: str

    @field_validator("check", "message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v

    @field_validator("value", "limit")
    @classmethod
    def _finite(cls, v: float | None) -> float | None:
        if v is not None and not math.isfinite(v):
            raise ValueError("must be a finite number")
        return v
