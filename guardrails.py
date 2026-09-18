"""Deterministic validation of raw LLM output. No I/O, no side effects.

The ONLY file allowed to accept or reject LLM output. Takes raw per-note
guesses + the original request battery, returns validated
List[DirectiveInterpretation] safe for both the client and optimizer.py.
"""

import math
from typing import List

from schemas import Battery, DirectiveInterpretation, RawDirective

ALLOWED_TYPES = frozenset(
    {
        "solar_reduction",
        "minimum_battery_reserve",
        "no_charge_window",
        "no_discharge_window",
        "max_grid_window",
        "no_op",
    }
)


def _clean_hours(value) -> list | None:
    """Unique ints 0..23 ascending, or None if unusable."""
    if not isinstance(value, list) or not value:
        return None
    cleaned: list = []
    for h in value:
        if isinstance(h, bool):
            return None
        if isinstance(h, float):
            if not h.is_integer():
                return None
            h = int(h)
        if not isinstance(h, int) or h < 0 or h > 23:
            return None
        if h not in cleaned:
            cleaned.append(h)
    if not cleaned:
        return None
    return sorted(cleaned)


def _finite_nonneg(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    if not math.isfinite(f) or f < 0:
        return None
    return f


def _to_no_op(note_index: int, reason: str) -> DirectiveInterpretation:
    return DirectiveInterpretation(
        note_index=note_index,
        applies=False,
        directive_type="no_op",
        structured_adjustment=None,
        explanation=f"Defaulted to no_op: {reason}",
    )


def validate_directives(
    raw: List[RawDirective], note_count: int, battery: Battery
) -> List[DirectiveInterpretation]:
    """Validate/repair raw LLM output. Never raises; always returns
    exactly note_count entries in ascending note_index order."""
    # Keep first occurrence per note_index, drop out-of-range.
    by_index: dict = {}
    for r in raw or []:
        try:
            idx = int(r.note_index)
        except Exception:
            continue
        if 0 <= idx < note_count and idx not in by_index:
            by_index[idx] = r

    out: List[DirectiveInterpretation] = []
    for i in range(note_count):
        r = by_index.get(i)
        if r is None:
            out.append(_to_no_op(i, "missing LLM output for this note."))
            continue
        out.append(_validate_one(i, r, battery))
    return out


def _validate_one(
    index: int, r: RawDirective, battery: Battery
) -> DirectiveInterpretation:
    dtype = r.directive_type if isinstance(r.directive_type, str) else ""
    if dtype not in ALLOWED_TYPES:
        return _to_no_op(index, f"unsupported directive_type {dtype!r}.")

    adj = r.structured_adjustment
    explanation = r.explanation if isinstance(r.explanation, str) else ""

    if dtype == "no_op":
        # no_op MUST be applies=false + null adjustment.
        return DirectiveInterpretation(
            note_index=index,
            applies=False,
            directive_type="no_op",
            structured_adjustment=None,
            explanation=explanation or "Note does not affect the energy schedule.",
        )

    if not isinstance(adj, dict):
        return _to_no_op(index, "structured_adjustment must be an object.")

    hours = _clean_hours(adj.get("hours"))
    if hours is None:
        return _to_no_op(index, "hours must be unique ints 0..23 ascending.")

    if dtype in ("no_charge_window", "no_discharge_window"):
        if set(adj.keys()) != {"hours"}:
            return _to_no_op(index, "unexpected keys for window directive.")
        return DirectiveInterpretation(
            note_index=index,
            applies=True,
            directive_type=dtype,  # type: ignore[arg-type]
            structured_adjustment={"hours": hours},
            explanation=explanation,
        )

    if dtype == "solar_reduction":
        if set(adj.keys()) != {"hours", "factor"}:
            return _to_no_op(index, "solar_reduction needs exactly hours+factor.")
        factor = adj.get("factor")
        if (
            isinstance(factor, bool)
            or not isinstance(factor, (int, float))
            or not math.isfinite(float(factor))
            or not 0 <= float(factor) <= 1
        ):
            return _to_no_op(index, "factor must be in [0, 1].")
        return DirectiveInterpretation(
            note_index=index,
            applies=True,
            directive_type=dtype,
            structured_adjustment={"hours": hours, "factor": float(factor)},
            explanation=explanation,
        )

    if dtype == "minimum_battery_reserve":
        if set(adj.keys()) != {"hours", "minimum_energy_kwh"}:
            return _to_no_op(index, "reserve needs exactly hours+minimum_energy_kwh.")
        val = _finite_nonneg(adj.get("minimum_energy_kwh"))
        if val is None or val > float(battery.capacity_kwh):
            return _to_no_op(index, "reserve must be within [0, capacity].")
        return DirectiveInterpretation(
            note_index=index,
            applies=True,
            directive_type=dtype,
            structured_adjustment={"hours": hours, "minimum_energy_kwh": val},
            explanation=explanation,
        )

    if dtype == "max_grid_window":
        if set(adj.keys()) != {"hours", "max_grid_kwh"}:
            return _to_no_op(index, "grid cap needs exactly hours+max_grid_kwh.")
        val = _finite_nonneg(adj.get("max_grid_kwh"))
        if val is None:
            return _to_no_op(index, "max_grid_kwh must be finite and >= 0.")
        return DirectiveInterpretation(
            note_index=index,
            applies=True,
            directive_type=dtype,
            structured_adjustment={"hours": hours, "max_grid_kwh": val},
            explanation=explanation,
        )

    return _to_no_op(index, "unrecognized directive.")  # unreachable
